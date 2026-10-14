// SPDX-License-Identifier: MIT
#pragma once

// Private runtime storage. Execution lives in the Slang dispatch module.
struct FluidSmokeCouplingStorage {
    FluidSmokeCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    std::uint32_t *converted{};
    ~FluidSmokeCouplingStorage() { release_managed(converted); }
};
