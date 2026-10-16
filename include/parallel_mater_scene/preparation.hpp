// SPDX-License-Identifier: MIT
#pragma once

#include "status.hpp"

#include "types.hpp"

#include <vector>

namespace parallel_mater {

inline constexpr std::uint32_t maximum_fluid_source_sites = 65'536U;

// Deterministic CPU-side preparation shared by device drivers and tools.
// These functions allocate no device resources and require no backend SDK.
[[nodiscard]] Status sample_rope_centerline(
    HostSpan<const Vec3> centerline, float spacing,
    std::vector<Vec3> &output) noexcept;

[[nodiscard]] Status build_soft_body_geometry(
    SoftBodyGeometrySource source, SoftBodyGeometry &output) noexcept;

[[nodiscard]] Status sample_fluid_source(
    ParticleSourceMesh mesh, std::vector<Vec3> &output) noexcept;

[[nodiscard]] Status sample_fluid_geometry(
    FluidGeometrySource source, std::vector<FluidParticle> &output) noexcept;

} // namespace parallel_mater
