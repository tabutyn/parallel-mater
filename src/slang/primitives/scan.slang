// SPDX-License-Identifier: MIT
// Shared small-array scans used by the simulation scheduler.

import device_intrinsics;
import graphics_resources;

[shader("compute")]
[numthreads(1, 1, 1)]
public void pm_exclusive_scan_u32(
    uniform PMBuffer<uint> input, uniform uint count, uniform PMBuffer<uint> output,
    uniform PMBuffer<uint> total PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmThreadIndex() != 0u) return;
    uint sum = 0u;
    for (uint index = 0u; index < count; ++index)
    {
        output[index] = sum;
        sum += input[index];
    }
    total[0] = sum;
}
