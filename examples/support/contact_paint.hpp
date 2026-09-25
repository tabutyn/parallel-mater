// SPDX-License-Identifier: MIT
#pragma once

#include "renderer_shared.hpp"

#include <parallel_mater/parallel_mater.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>

namespace parallel_mater::gallery {

// Example-only persistent paint. Fluid particles near authored render
// triangles wet their UVs; the installed physics API owns no textures.
struct PaintBinding {
    RigidBodyState state{};
    const optix_shared::Vertex *vertices{};
    const uint3 *triangles{};
    std::uint32_t triangle_count{};
    std::uint32_t *pixels{};
    std::uint32_t width{};
    std::uint32_t height{};
};

[[nodiscard]] cudaError_t apply_particle_paint(
    const Vec3 *positions, std::uint32_t particle_count,
    const PaintBinding *bindings, std::uint32_t binding_count,
    float particle_radius, cudaStream_t stream = nullptr) noexcept;

} // namespace parallel_mater::gallery
