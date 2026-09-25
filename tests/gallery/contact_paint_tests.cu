// SPDX-License-Identifier: MIT
#include "contact_paint.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>

int main() {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    const std::array<optix_shared::Vertex, 3> vertices{{
        {{0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, {0.99F, 0.5F}},
        {{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, {0.99F, 0.5F}},
        {{0.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.0F}, {0.99F, 0.5F}}}};
    const uint3 triangle{0U, 1U, 2U};
    const Vec3 particle{0.2F, 0.03F, 0.2F};
    optix_shared::Vertex *gpu_vertices = nullptr;
    uint3 *gpu_triangles = nullptr;
    std::uint32_t *gpu_pixels = nullptr;
    Vec3 *gpu_positions = nullptr;
    PaintBinding *gpu_bindings = nullptr;
    constexpr std::uint32_t width = 64U, height = 64U;
    bool okay =
        cudaMalloc(&gpu_vertices, sizeof(vertices)) == cudaSuccess &&
        cudaMalloc(&gpu_triangles, sizeof(triangle)) == cudaSuccess &&
        cudaMalloc(&gpu_pixels, width * height * sizeof(std::uint32_t)) ==
            cudaSuccess &&
        cudaMalloc(&gpu_positions, sizeof(particle)) == cudaSuccess &&
        cudaMalloc(&gpu_bindings, sizeof(PaintBinding)) == cudaSuccess;
    if (okay) {
        const PaintBinding binding{{}, gpu_vertices, gpu_triangles,
                                   1U, gpu_pixels, width, height};
        okay = cudaMemcpy(gpu_vertices, vertices.data(), sizeof(vertices),
                          cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(gpu_triangles, &triangle, sizeof(triangle),
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemset(gpu_pixels, 0, width * height * sizeof(std::uint32_t)) ==
                cudaSuccess &&
            cudaMemcpy(gpu_positions, &particle, sizeof(particle),
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(gpu_bindings, &binding, sizeof(binding),
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            apply_particle_paint(gpu_positions, 1U, gpu_bindings, 1U, 0.03F) ==
                cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    }
    std::array<std::uint32_t, width * height> pixels{};
    if (okay)
        okay = cudaMemcpy(pixels.data(), gpu_pixels, sizeof(pixels),
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    okay = okay && pixels[32U * width + 63U] == 2U &&
        pixels[32U * width] == 0U &&
        pixels[31U * width + 63U] == 0U &&
        pixels[8U * width + 8U] == 0U;
    if (okay) {
        Vec3 opposite = particle;
        opposite.y = -0.03F;
        okay = cudaMemcpy(gpu_positions, &opposite, sizeof(opposite),
                          cudaMemcpyHostToDevice) == cudaSuccess &&
            apply_particle_paint(gpu_positions, 1U, gpu_bindings, 1U, 0.03F) ==
                cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
            cudaMemcpy(pixels.data(), gpu_pixels, sizeof(pixels),
                       cudaMemcpyDeviceToHost) == cudaSuccess &&
            pixels[32U * width + 63U] == 3U;
    }
    cudaFree(gpu_bindings);
    cudaFree(gpu_positions);
    cudaFree(gpu_pixels);
    cudaFree(gpu_triangles);
    cudaFree(gpu_vertices);
    if (!okay) {
        std::cerr << "particle paint did not persist at the authored UV seam\n";
        return 1;
    }
    return 0;
}
