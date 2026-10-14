// SPDX-License-Identifier: MIT
// Portable dispatch and atomic operations used by shared physics kernels.

import types;

#if defined(PM_TARGET_VULKAN) || defined(PM_TARGET_METAL)
public in uint3 pmVulkanGroupId : SV_GroupID;
public in uint3 pmVulkanGroupThreadId : SV_GroupThreadID;
public in uint3 pmVulkanDispatchThreadId : SV_DispatchThreadID;
uniform uint3 pmDispatchGroupCount;
uniform uint pmDispatchGroupSizeX;

public uint3 pmBlockIndex() { return pmVulkanGroupId; }
public uint3 pmGroupThreadIndex() { return pmVulkanGroupThreadId; }
public uint3 pmGroupSize() { return uint3(pmDispatchGroupSizeX, 1u, 1u); }

public uint pmAtomicAdd(uint* value, uint addend)
{
    uint original;
    InterlockedAdd(*value, addend, original);
    return original;
}
public uint64_t pmAtomicAdd(uint64_t* value, uint64_t addend)
{
    uint64_t original;
    InterlockedAdd(*value, addend, original);
    return original;
}
public float pmAtomicAdd(float* value, float addend)
{
    float original;
    InterlockedAdd(*value, addend, original);
    return original;
}
public uint pmAtomicMax(uint* value, uint candidate)
{
    uint original;
    InterlockedMax(*value, candidate, original);
    return original;
}
public uint pmAtomicMin(uint* value, uint candidate)
{
    uint original;
    InterlockedMin(*value, candidate, original);
    return original;
}
public uint pmAtomicOr(uint* value, uint candidate)
{
    uint original;
    InterlockedOr(*value, candidate, original);
    return original;
}
public uint64_t pmAtomicOr(uint64_t* value, uint64_t candidate)
{
    uint64_t original;
    InterlockedOr(*value, candidate, original);
    return original;
}
public uint pmAtomicExchange(uint* value, uint replacement)
{
    uint original;
    InterlockedExchange(*value, replacement, original);
    return original;
}
public void pmSyncWarp(uint) { GroupMemoryBarrierWithGroupSync(); }
public uint3 pmGridDim() { return pmDispatchGroupCount; }

// Vulkan has no in-dispatch grid barrier. The Vulkan executor splits logical
// schedules at each grid dependency; this remains a group barrier in a pass.
public void pmGridBarrier() { DeviceMemoryBarrierWithGroupSync(); }
#else
public uint3 pmBlockIndex() { return cudaBlockIdx(); }
public uint3 pmGroupThreadIndex() { return cudaThreadIdx(); }
public uint3 pmGroupSize() { return cudaBlockDim(); }

public uint pmThreadIndex()
{
    return pmBlockIndex().x * pmGroupSize().x + pmGroupThreadIndex().x;
}

__target_intrinsic(cuda, "make_uint3(gridDim.x, gridDim.y, gridDim.z)")
public uint3 pmGridDim();

__target_intrinsic(cuda, "atomicAdd") public uint pmAtomicAdd(uint* value, uint addend);
__target_intrinsic(cuda, "atomicAdd") public uint64_t pmAtomicAdd(
    uint64_t* value, uint64_t addend);
__target_intrinsic(cuda, "atomicAdd") public float pmAtomicAdd(float* value, float addend);
__target_intrinsic(cuda, "atomicMax") public uint pmAtomicMax(uint* value, uint candidate);
__target_intrinsic(cuda, "atomicMin") public uint pmAtomicMin(uint* value, uint candidate);
__target_intrinsic(cuda, "atomicOr") public uint pmAtomicOr(uint* value, uint candidate);
__target_intrinsic(cuda, "atomicOr") public uint64_t pmAtomicOr(
    uint64_t* value, uint64_t candidate);
__target_intrinsic(cuda, "atomicExch") public uint pmAtomicExchange(uint* value, uint replacement);
__target_intrinsic(cuda, "__syncwarp") public void pmSyncWarp(uint mask);
__target_intrinsic(cuda, "pm_cuda_grid_barrier()")
public void pmNativeGridBarrier();

public void pmGridBarrier()
{
    __requirePrelude("struct PmCudaGridWorkspace { unsigned int size; unsigned int barrier; };\n"
        "__device__ __forceinline__ void pm_cuda_grid_barrier() {\n"
        "  unsigned int lo, hi;\n"
        "  asm volatile(\"mov.u32 %0, %%envreg2;\" : \"=r\"(lo));\n"
        "  asm volatile(\"mov.u32 %0, %%envreg1;\" : \"=r\"(hi));\n"
        "  unsigned long long address = (static_cast<unsigned long long>(hi) << 32) | lo;\n"
        "  volatile unsigned int* arrived = &reinterpret_cast<PmCudaGridWorkspace*>(address)->barrier;\n"
        "  __syncthreads();\n"
        "  unsigned int old_arrive = 0;\n"
        "  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {\n"
        "    const unsigned int expected = gridDim.x * gridDim.y * gridDim.z;\n"
        "    const bool grid_master = blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;\n"
        "    const unsigned int increment = grid_master ? 0x80000000u - (expected - 1u) : 1u;\n"
        "    __threadfence();\n"
        "    old_arrive = atomicAdd(const_cast<unsigned int*>(arrived), increment);\n"
        "    while (((old_arrive ^ *arrived) & 0x80000000u) == 0u) {}\n"
        "    __threadfence();\n"
        "  }\n"
        "  __syncthreads();\n"
        "}\n");
    pmNativeGridBarrier();
}
#endif

#if defined(PM_TARGET_VULKAN) || defined(PM_TARGET_METAL)
public uint pmThreadIndex() { return pmVulkanDispatchThreadId.x; }
#endif

public void pmGroupBarrier() { GroupMemoryBarrierWithGroupSync(); }
