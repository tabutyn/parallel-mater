// SPDX-License-Identifier: MIT
// Entry-point system values cannot be read implicitly by DXIL library code.
// Keep the initialization identical for every shared compute schedule.
#ifndef PM_DISPATCH_ENTRY_SLANG
#define PM_DISPATCH_ENTRY_SLANG
#ifndef PM_TARGET_D3D12
#define PM_TARGET_D3D12 0
#endif
#if PM_TARGET_D3D12
#define PM_D3D12_DISPATCH_PARAMETER \
    , uint3 pmD3D12DispatchThreadId : SV_DispatchThreadID
#define PM_D3D12_DISPATCH_ARGUMENT , pmD3D12DispatchThreadId
#define PM_D3D12_DISPATCH_PROLOGUE() \
    pmSetD3D12DispatchThreadId(pmD3D12DispatchThreadId)
#else
#define PM_D3D12_DISPATCH_PARAMETER
#define PM_D3D12_DISPATCH_ARGUMENT
#define PM_D3D12_DISPATCH_PROLOGUE()
#endif
#endif
