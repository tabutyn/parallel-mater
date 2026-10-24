// SPDX-License-Identifier: MIT
// Shared bounded scans used by the simulation scheduler. One workgroup scans
// an arbitrary number of values by assigning each lane one contiguous chunk;
// this keeps the scan at one dispatch without serializing the whole array on a
// single GPU lane.

import device_intrinsics;
import graphics_resources;

groupshared uint pmScanLaneTotals[256];

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_exclusive_scan_u32(
    uniform PMBuffer<uint> input, uniform uint count, uniform PMBuffer<uint> output,
    uniform PMBuffer<uint> total PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint lane = pmGroupThreadIndex().x;
    uint chunk = (count + 255u) / 256u;
    uint begin = min(count, lane * chunk);
    uint end = min(count, begin + chunk);
    uint sum = 0u;
    for (uint index = begin; index < end; ++index)
    {
        output[index] = sum;
        sum += input[index];
    }
    pmScanLaneTotals[lane] = sum;
    pmGroupBarrier();

    if (lane == 0u)
    {
        uint base = 0u;
        for (uint index = 0u; index < 256u; ++index)
        {
            uint laneTotal = pmScanLaneTotals[index];
            pmScanLaneTotals[index] = base;
            base += laneTotal;
        }
        total[0] = base;
    }
    pmGroupBarrier();

    uint base = pmScanLaneTotals[lane];
    for (uint index = begin; index < end; ++index)
        output[index] += base;
}
