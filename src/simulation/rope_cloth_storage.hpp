// SPDX-License-Identifier: MIT
#pragma once

// Private runtime storage. Physics execution lives in the generated Slang
// dispatch module.
struct RopeClothCouplingStorage {
    RopeClothCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
};
