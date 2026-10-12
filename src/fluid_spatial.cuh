// SPDX-License-Identifier: MIT
#pragma once

// Transitional device helpers used by native coupling code. The production
// fluid kernels use the equivalent functions in fluid.slang.
__host__ __device__ std::uint64_t fluid_cell_key(
    int x, int y, int z) noexcept {
    x = max(-k_fluid_cell_bias, min(k_fluid_cell_bias - 1, x));
    y = max(-k_fluid_cell_bias, min(k_fluid_cell_bias - 1, y));
    z = max(-k_fluid_cell_bias, min(k_fluid_cell_bias - 1, z));
    return (static_cast<std::uint64_t>(x + k_fluid_cell_bias) << 42U) |
           (static_cast<std::uint64_t>(y + k_fluid_cell_bias) << 21U) |
           static_cast<std::uint64_t>(z + k_fluid_cell_bias);
}

__device__ std::uint32_t fluid_lower_bound(const std::uint64_t *keys,
                                           std::uint32_t size,
                                           std::uint64_t key) noexcept {
    std::uint32_t lo = 0U, hi = size;
    while (lo < hi) {
        const std::uint32_t middle = lo + (hi - lo) / 2U;
        if (keys[middle] < key) lo = middle + 1U;
        else hi = middle;
    }
    return lo;
}
