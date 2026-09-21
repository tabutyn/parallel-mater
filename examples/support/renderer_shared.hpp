// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime.h>
#include <optix.h>

namespace parallel_mater::gallery::optix_shared {

struct Vertex {
    float3 position{};
    float3 normal{};
};

struct LaunchParameters {
    uchar4 *image{};
    unsigned int width{};
    unsigned int height{};
    OptixTraversableHandle scene{};
    float3 eye{};
    float3 camera_u{};
    float3 camera_v{};
    float3 camera_w{};
};

struct HitData {
    const Vertex *vertices{};
    const uint3 *triangles{};
    float3 base_color{};
    unsigned int checkerboard{};
};

} // namespace parallel_mater::gallery::optix_shared
