// SPDX-License-Identifier: MIT
// Shared small-array scans used by the simulation scheduler.

import device_intrinsics;

[shader("compute")]
[numthreads(1, 1, 1)]
public void pm_exclusive_scan_u32(
    uniform uint* input, uniform uint count, uniform uint* output,
    uniform uint* total)
{
    if (pmThreadIndex() != 0u) return;
    uint sum = 0u;
    for (uint index = 0u; index < count; ++index)
    {
        output[index] = sum;
        sum += input[index];
    }
    *total = sum;
}
