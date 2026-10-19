// SPDX-License-Identifier: MIT
#pragma once

// Private runtime storage. Execution lives in the Slang dispatch module.
struct FluidRopeCouplingStorage {
    FluidRopeCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    std::uint32_t *contribution_nodes{};
    Vec3 *contribution_impulses{};
    std::uint32_t *contact_count{};
    float *maximum_penetration{};
    void release() noexcept {
        release_managed(contribution_nodes);
        release_managed(contribution_impulses);
        release_managed(contact_count);
        release_managed(maximum_penetration);
    }
    ~FluidRopeCouplingStorage() { release(); }
};
