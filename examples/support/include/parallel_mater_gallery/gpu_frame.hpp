// SPDX-License-Identifier: MIT
#pragma once
#include <parallel_mater/parallel_mater.h>
#include <vector>
#include <cstdint>

namespace parallel_mater::gallery {

// Borrowed public output resources, valid until the CommandScene is reset or
// destroyed. Rendering consumes API exports, never private simulation storage.
struct GpuSceneBuffers {
    PmDevice device{};
    PmQueue queue{};
    PmBuffer output{};
    PmDeviceSize rigid_offset{};
    PmDeviceSize previous_rigid_offset{};
    PmDeviceSize fluid_offset{};
    std::uint32_t fluid_capacity{};
    std::vector<PmDeviceSize> cloth_surface_offsets;
};

// An adapter-owned image shared with native presentation. A zero handle means
// the renderer returned a CPU image instead. CPU pixels then contain only UI
// when a native image is present.
struct NativeRenderImage {
    PmNativeBackend backend{};
    PmNativeHandle handle{};
};
} // namespace parallel_mater::gallery
