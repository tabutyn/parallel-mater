// SPDX-License-Identifier: MIT
#pragma once

// Included by the transitional scheduler after public coupling types and
// managed-allocation helpers are in scope. Execution lives in Slang dispatch.
struct FluidRopeCouplingStorage {
    FluidRopeCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    Vec3 *node_impulses{};
    std::uint32_t *contact_count{};
    float *maximum_penetration{};
    void release() noexcept {
        release_managed(node_impulses);
        release_managed(contact_count);
        release_managed(maximum_penetration);
    }
    ~FluidRopeCouplingStorage() { release(); }
};
