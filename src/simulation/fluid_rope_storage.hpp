// SPDX-License-Identifier: MIT
#pragma once

// Private runtime storage. Execution lives in the Slang dispatch module.
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
