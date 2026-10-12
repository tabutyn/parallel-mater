// SPDX-License-Identifier: MIT
#pragma once

// Included by the transitional scheduler after public coupling types are in
// scope. Physics execution lives in the generated Slang dispatch module.
struct RopeClothCouplingStorage {
    RopeClothCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
};
