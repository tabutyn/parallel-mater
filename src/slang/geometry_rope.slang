// SPDX-License-Identifier: MIT
// Slang mirror of src/geometry_rope.cuh. Core geometry/solver kernels are
// defined below the shared rope ABI helpers as this migration advances.

import cuda_common;

public int rope_anchor_body(RopeData rope, uint node, int first, int last)
{
    return node == 0u ? first : (node + 1u == rope.count ? last : -1);
}

public bool rope_soft_anchor(RopeData rope, uint node)
{
    return (node == 0u && rope.soft_first) ||
           (node + 1u == rope.count && rope.soft_last);
}

public uint rope_soft_end(RopeData rope, uint node)
{
    return node == 0u ? 0u : 1u;
}
