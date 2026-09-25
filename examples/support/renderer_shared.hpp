// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime.h>
#include <optix.h>

namespace parallel_mater::gallery::optix_shared {

struct Vertex {
    float3 position{};
    float3 normal{};
    float2 uv{};
};

struct FluidSurfaceGrid {
    float3 minimum{};
    float3 cell_size{};
    uint3 dimensions{};
    float support_radius{};
    float particle_radius{};
};

struct FluidSurfaceView {
    const float *values{};
    FluidSurfaceGrid grid{};
};

struct LaunchParameters {
    uchar4 *image{};
    float *depth{};
    float *rigid_depth{};
    unsigned int width{};
    unsigned int height{};
    OptixTraversableHandle scene{};
    float3 eye{};
    float3 camera_u{};
    float3 camera_v{};
    float3 camera_w{};
    FluidSurfaceView fluid{};
};

struct HitData {
    const Vertex *vertices{};
    const uint3 *triangles{};
    const unsigned int *paint_pixels{};
    unsigned int paint_width{};
    unsigned int paint_height{};
    float3 base_color{};
    unsigned int checkerboard{};
};

} // namespace parallel_mater::gallery::optix_shared
