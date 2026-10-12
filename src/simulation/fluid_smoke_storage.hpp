// SPDX-License-Identifier: MIT
#pragma once

// Included by the transitional scheduler after public coupling types and
// managed-allocation helpers are in scope. Execution lives in Slang dispatch.
struct FluidSmokeCouplingStorage {
    FluidSmokeCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    std::uint32_t *converted{};
    ~FluidSmokeCouplingStorage() { release_managed(converted); }
};
