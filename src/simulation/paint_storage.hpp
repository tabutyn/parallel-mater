// SPDX-License-Identifier: MIT
#pragma once
#include "simulation/runtime.hpp"

namespace parallel_mater::simulation {
// Shared host resource layout; the paint equations live in Slang.
struct PaintFieldResource {
    PaintFieldOptions options{};
    Vec2 *uvs{};
    std::uint32_t *pixels{};
    std::uint32_t generation{1U};
    bool alive{};
};
struct PaintRuleResource {
    PaintRuleOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
};
} // namespace parallel_mater::simulation
