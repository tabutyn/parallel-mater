// SPDX-License-Identifier: MIT
#include "contact_paint.hpp"

#include <cuda_runtime.h>

#include <cmath>

namespace parallel_mater::gallery {
namespace {

__device__ float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 subtract(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 multiply(float3 a, float scale) {
    return make_float3(a.x * scale, a.y * scale, a.z * scale);
}
__device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}
__device__ float3 inverse_rotate(Quaternion q, float3 value) {
    const float3 axis = make_float3(-q.x, -q.y, -q.z);
    const float3 twice = multiply(make_float3(
        axis.y * value.z - axis.z * value.y,
        axis.z * value.x - axis.x * value.z,
        axis.x * value.y - axis.y * value.x), 2.0F);
    return add(value, add(multiply(twice, q.w), make_float3(
        axis.y * twice.z - axis.z * twice.y,
        axis.z * twice.x - axis.x * twice.z,
        axis.x * twice.y - axis.y * twice.x)));
}

// Closest point and barycentrics from the vertex/edge/face Voronoi regions.
__device__ float3 closest_triangle(float3 p, float3 a, float3 b, float3 c,
                                   float3 &weights) {
    const float3 ab = subtract(b, a), ac = subtract(c, a);
    const float3 ap = subtract(p, a);
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0F && d2 <= 0.0F) {
        weights = make_float3(1.0F, 0.0F, 0.0F); return a;
    }
    const float3 bp = subtract(p, b);
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0F && d4 <= d3) {
        weights = make_float3(0.0F, 1.0F, 0.0F); return b;
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0F && d1 >= 0.0F && d3 <= 0.0F) {
        const float v = d1 / (d1 - d3);
        weights = make_float3(1.0F - v, v, 0.0F);
        return add(a, multiply(ab, v));
    }
    const float3 cp = subtract(p, c);
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0F && d5 <= d6) {
        weights = make_float3(0.0F, 0.0F, 1.0F); return c;
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0F && d2 >= 0.0F && d6 <= 0.0F) {
        const float w = d2 / (d2 - d6);
        weights = make_float3(1.0F - w, 0.0F, w);
        return add(a, multiply(ac, w));
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0F && d4 - d3 >= 0.0F && d5 - d6 >= 0.0F) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        weights = make_float3(0.0F, 1.0F - w, w);
        return add(b, multiply(subtract(c, b), w));
    }
    const float denominator = va + vb + vc;
    if (fabsf(denominator) < 1.0e-12F) {
        weights = make_float3(1.0F, 0.0F, 0.0F);
        return a;
    }
    const float inverse = 1.0F / denominator;
    const float v = vb * inverse, w = vc * inverse;
    weights = make_float3(1.0F - v - w, v, w);
    return add(a, add(multiply(ab, v), multiply(ac, w)));
}

__global__ void paint_particles(const Vec3 *positions,
                                std::uint32_t particle_count,
                                const PaintBinding *bindings,
                                std::uint32_t binding_count,
                                float maximum_distance_squared,
                                float particle_radius) {
    const std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= particle_count) return;
    const Vec3 point = positions[item];
    for (std::uint32_t binding_index = 0U; binding_index < binding_count;
         ++binding_index) {
        const PaintBinding binding = bindings[binding_index];
        if (binding.pixels == nullptr) continue;
        const Vec3 center = binding.state.position;
        const float3 local = inverse_rotate(binding.state.orientation,
            subtract(make_float3(point.x, point.y, point.z),
                     make_float3(center.x, center.y, center.z)));
        float best_distance = maximum_distance_squared;
        float2 best_uv{};
        std::uint32_t best_side = 0U;
        int best_radius = 1;
        for (std::uint32_t triangle_index = 0U;
             triangle_index < binding.triangle_count; ++triangle_index) {
            const uint3 triangle = binding.triangles[triangle_index];
            const optix_shared::Vertex a = binding.vertices[triangle.x];
            const optix_shared::Vertex b = binding.vertices[triangle.y];
            const optix_shared::Vertex c = binding.vertices[triangle.z];
            float3 weights{};
            const float3 nearest = closest_triangle(
                local, a.position, b.position, c.position, weights);
            const float3 delta = subtract(local, nearest);
            const float distance = dot(delta, delta);
            if (distance >= best_distance) continue;
            best_distance = distance;
            best_uv = make_float2(
                weights.x * a.uv.x + weights.y * b.uv.x + weights.z * c.uv.x,
                weights.x * a.uv.y + weights.y * b.uv.y + weights.z * c.uv.y);
            const float3 face = cross(subtract(b.position, a.position),
                                      subtract(c.position, a.position));
            best_side = dot(face, delta) >= 0.0F ? 1U : 2U;
            const float world_area_twice = sqrtf(dot(face, face));
            const float uv_area_twice = fabsf(
                (b.uv.x - a.uv.x) * (c.uv.y - a.uv.y) -
                (b.uv.y - a.uv.y) * (c.uv.x - a.uv.x));
            // A single physical particle should paint roughly the same
            // surface area regardless of mesh size or UV texture density.
            const float pixels_per_world = sqrtf(
                uv_area_twice / fmaxf(world_area_twice, 1.0e-12F) *
                binding.width * binding.height);
            best_radius = max(1, min(8, __float2int_rn(
                1.5F * particle_radius * pixels_per_world)));
        }
        if (best_side == 0U) continue;
        const int width = static_cast<int>(binding.width);
        const int height = static_cast<int>(binding.height);
        const int center_x = static_cast<int>(floorf(best_uv.x * width));
        const int center_y = static_cast<int>(floorf(best_uv.y * height));
        for (int dy = -best_radius; dy <= best_radius; ++dy) {
            const int y = center_y + dy;
            if (y < 0 || y >= height) continue;
            for (int dx = -best_radius; dx <= best_radius; ++dx) {
                if (dx * dx + dy * dy > best_radius * best_radius) continue;
                int x = (center_x + dx) % width;
                if (x < 0) x += width;
                atomicOr(binding.pixels + y * width + x, best_side);
            }
        }
    }
}

} // namespace

cudaError_t apply_particle_paint(const Vec3 *positions,
                                 std::uint32_t particle_count,
                                 const PaintBinding *bindings,
                                 std::uint32_t binding_count,
                                 float particle_radius,
                                 cudaStream_t stream) noexcept {
    if (particle_count == 0U || binding_count == 0U) return cudaSuccess;
    const float reach = particle_radius + 0.025F;
    paint_particles<<<(particle_count + 127U) / 128U, 128U, 0, stream>>>(
        positions, particle_count, bindings, binding_count, reach * reach,
        particle_radius);
    return cudaPeekAtLastError();
}

} // namespace parallel_mater::gallery
