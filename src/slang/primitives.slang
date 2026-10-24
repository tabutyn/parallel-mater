// SPDX-License-Identifier: MIT
#include "dispatch_shape.slangh"
// Scheduler-visible collection of shared scan, compaction, and sort kernels.

#ifndef PM_TARGET_D3D12
#define PM_TARGET_D3D12 0
#endif
#if PM_TARGET_D3D12
#define PM_D3D12_DISPATCH_PARAMETER \
    , uint3 pmD3D12DispatchThreadId : SV_DispatchThreadID
#define PM_D3D12_DISPATCH_PROLOGUE() \
    pmSetD3D12DispatchThreadId(pmD3D12DispatchThreadId)
#else
#define PM_D3D12_DISPATCH_PARAMETER
#define PM_D3D12_DISPATCH_PROLOGUE()
#endif

#include "primitives/scan.slang"
#include "primitives/compaction.slang"
#include "primitives/radix_sort.slang"
#include "primitives/spatial_build.slang"
