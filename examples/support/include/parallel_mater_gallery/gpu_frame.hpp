// SPDX-License-Identifier: MIT
#pragma once
#include <parallel_mater/parallel_mater.h>
#include <vector>
#include <cstdint>

namespace parallel_mater::gallery {

// Borrowed public output resources, valid until the CommandScene is reset or
// destroyed. Finish rendering before the next step overwrites these exports.
// Rendering consumes API exports, never private simulation storage.
struct GpuSceneBuffers {
    PmDevice device{};
    PmQueue queue{};
    PmBuffer output{};
    struct Region { PmDeviceSize offset{}, size{}; };
    Region rigid, previous_rigid, fluid, smoke;
    std::uint32_t fluid_capacity{};
    std::vector<Region> cloth_surfaces, soft_surfaces, ropes, paint;
};

// An adapter-owned image shared with native presentation. A zero handle means
// the renderer returned a CPU image instead. CPU pixels then contain only UI
// when a native image is present.
struct NativeRenderImage {
    PmNativeBackend backend{};
    PmNativeHandle handle{};
};
} // namespace parallel_mater::gallery
