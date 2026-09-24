// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime.h>
#include <optix.h>

namespace parallel_mater::gallery::optix_shared {

struct Vertex {
    float3 position{};
    float3 normal{};
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
    float3 base_color{};
    unsigned int checkerboard{};
};

} // namespace parallel_mater::gallery::optix_shared
