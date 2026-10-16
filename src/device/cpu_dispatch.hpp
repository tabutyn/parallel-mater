// SPDX-License-Identifier: MIT
#pragma once

#include "device.hpp"
#include <slang-cpp-prelude.h>

#include <cstdint>

// Generated entry points are an implementation detail of this driver.  The
// public driver ABI exports only the four pm_driver* negotiation symbols.
#undef SLANG_PRELUDE_EXPORT
#define SLANG_PRELUDE_EXPORT extern "C"

// Slang's C++ target calls these intrinsics from the generated kernels.  The
// adapter owns the indices, so the Slang modules remain backend-neutral.
static thread_local uint3 pm_cpu_block{};
static thread_local uint3 pm_cpu_thread{};
static thread_local uint3 pm_cpu_block_size{1U, 1U, 1U};
static thread_local uint3 pm_cpu_grid{1U, 1U, 1U};

static inline uint3 pm_cpu_block_index() noexcept { return pm_cpu_block; }
static inline uint3 pm_cpu_group_thread_index() noexcept { return pm_cpu_thread; }
static inline uint3 pm_cpu_group_size() noexcept { return pm_cpu_block_size; }
static inline uint3 pm_cpu_grid_dim() noexcept { return pm_cpu_grid; }

namespace parallel_mater::device {

using CpuKernel = void (*)(ComputeThreadVaryingInput *, void *, void *);

inline void dispatch_cpu(DispatchSize grid, DispatchSize block, CpuKernel kernel,
                         void *parameters) noexcept {
    pm_cpu_grid = {grid.x, grid.y, grid.z};
    pm_cpu_block_size = {block.x, block.y, block.z};
    ComputeThreadVaryingInput varying{};
    for (std::uint32_t gz = 0U; gz < grid.z; ++gz)
        for (std::uint32_t gy = 0U; gy < grid.y; ++gy)
            for (std::uint32_t gx = 0U; gx < grid.x; ++gx) {
                pm_cpu_block = {gx, gy, gz};
                varying.groupID = pm_cpu_block;
                for (std::uint32_t tz = 0U; tz < block.z; ++tz)
                    for (std::uint32_t ty = 0U; ty < block.y; ++ty)
                        for (std::uint32_t tx = 0U; tx < block.x; ++tx) {
                            pm_cpu_thread = {tx, ty, tz};
                            varying.groupThreadID = pm_cpu_thread;
                            kernel(&varying, parameters, nullptr);
                        }
            }
}

// Cooperative kernels are written so lane zero walks every item when the
// group size is one.  Running one such lane per requested group makes barriers
// ordinary sequencing points and retains the shared Slang schedule.
inline void dispatch_cpu_serial_groups(DispatchSize grid, CpuKernel kernel,
                                       void *parameters) noexcept {
    dispatch_cpu(grid, DispatchSize{1U}, kernel, parameters);
}

inline void dispatch_cpu_serial(CpuKernel kernel, void *parameters) noexcept {
    dispatch_cpu(DispatchSize{1U}, DispatchSize{1U}, kernel, parameters);
}

} // namespace parallel_mater::device
