// SPDX-License-Identifier: MIT
// Portable dispatch and atomic operations used by shared physics kernels.

import types;
import graphics_resources;

#if defined(PM_VULKAN_VARIABLE_GROUP_SIZE)
[[vk::constant_id(0)]] public const uint pmVulkanWorkgroupSizeX = 256;
#endif

#ifndef PM_TARGET_D3D12
#define PM_TARGET_D3D12 0
#endif

#if defined(PM_TARGET_CPU)
__target_intrinsic(cpp, "pm_cpu_block_index")
public uint3 pmBlockIndex();
__target_intrinsic(cpp, "pm_cpu_group_thread_index")
public uint3 pmGroupThreadIndex();
__target_intrinsic(cpp, "pm_cpu_group_size")
public uint3 pmGroupSize();
__target_intrinsic(cpp, "pm_cpu_grid_dim")
public uint3 pmGridDim();

public uint pmThreadIndex()
{
    return pmBlockIndex().x * pmGroupSize().x + pmGroupThreadIndex().x;
}

// CPU dispatch executes one lane at a time. Preserve atomic return values
// without requesting GPU-only atomic instructions from Slang's C++ target.
public uint pmAtomicAdd(uint* value, uint addend)
{
    uint original = *value; *value += addend; return original;
}
public uint64_t pmAtomicAdd(uint64_t* value, uint64_t addend)
{
    uint64_t original = *value; *value += addend; return original;
}
public float pmAtomicAdd(float* value, float addend)
{
    float original = *value; *value += addend; return original;
}
public uint pmAtomicMax(uint* value, uint candidate)
{
    uint original = *value; *value = max(original, candidate); return original;
}
public uint pmAtomicMin(uint* value, uint candidate)
{
    uint original = *value; *value = min(original, candidate); return original;
}
public uint pmAtomicOr(uint* value, uint candidate)
{
    uint original = *value; *value |= candidate; return original;
}
public uint64_t pmAtomicOr(uint64_t* value, uint64_t candidate)
{
    uint64_t original = *value; *value |= candidate; return original;
}
public uint pmAtomicExchange(uint* value, uint replacement)
{
    uint original = *value; *value = replacement; return original;
}
public void pmAtomicAddVec3(PMBuffer<Vec3> value, Vec3 addend)
{
    pmAtomicAdd(&value[0].x, addend.x);
    pmAtomicAdd(&value[0].y, addend.y);
    pmAtomicAdd(&value[0].z, addend.z);
}
public uint pmAtomicMaxPositiveFloat(PMBuffer<float> value, float candidate)
{
    return pmAtomicMax(reinterpret<uint*>(value), asuint(candidate));
}
public uint pmAtomicMinPositiveFloat(PMBuffer<float> value, float candidate)
{
    return pmAtomicMin(reinterpret<uint*>(value), asuint(candidate));
}
public void pmSyncWarp(uint) {}
public void pmGridBarrier() {}
public void pmGroupBarrier() {}
#elif defined(PM_TARGET_VULKAN)
public in uint3 pmGroupId : SV_GroupID;
public in uint3 pmGroupThreadId : SV_GroupThreadID;
public in uint3 pmDispatchThreadId : SV_DispatchThreadID;
#elif PM_TARGET_D3D12
static uint3 pmGroupId;
static uint3 pmGroupThreadId;
static uint3 pmDispatchThreadId;

public void pmSetD3D12ThreadIndices(uint3 groupId, uint3 groupThreadId,
    uint3 dispatchThreadId)
{
    pmGroupId = groupId;
    pmGroupThreadId = groupThreadId;
    pmDispatchThreadId = dispatchThreadId;
}
#endif

#if defined(PM_TARGET_CPU)
// The host adapter supplies dispatch indices through the C++ target intrinsics
// above and executes kernels serially, so no second implementation is needed.
#elif defined(PM_TARGET_VULKAN) || PM_TARGET_D3D12
uniform uint3 pmDispatchGroupCount;
uniform uint pmDispatchGroupSizeX;

public uint3 pmBlockIndex() { return pmGroupId; }
public uint3 pmGroupThreadIndex() { return pmGroupThreadId; }
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
#if defined(PM_TARGET_VULKAN)
    uint* bits = reinterpret<uint*>(value);
    uint expected;
    InterlockedAdd(*bits, 0u, expected);
    for (;;)
    {
        uint original;
        InterlockedCompareExchange(*bits, expected,
            asuint(asfloat(expected) + addend), original);
        if (original == expected) return asfloat(original);
        expected = original;
    }
#else
    float original;
    InterlockedAdd(*value, addend, original);
    return original;
#endif
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

#if PM_TARGET_D3D12
public uint pmAtomicAdd(PMBuffer<uint> value, uint addend)
{
    RWByteAddressBuffer resource = pmGraphicsResource(value);
    uint original;
    resource.InterlockedAdd(value.byte_offset, addend, original);
    return original;
}
public uint64_t pmAtomicAdd(PMBuffer<uint64_t> value, uint64_t addend)
{
    RWByteAddressBuffer resource = pmGraphicsResource(value);
    uint64_t original;
    resource.InterlockedAdd64(
        value.byte_offset, addend, original);
    return original;
}
public float pmAtomicAdd(PMBuffer<float> value, float addend)
{
    RWByteAddressBuffer resource = pmGraphicsResource(value);
    uint address = value.byte_offset;
    uint expected = resource.Load<uint>(address);
    for (;;)
    {
        uint replacement = asuint(asfloat(expected) + addend);
        uint original;
        resource.InterlockedCompareExchange(
            address, expected, replacement, original);
        if (original == expected) return asfloat(original);
        expected = original;
    }
}
public uint pmAtomicMax(PMBuffer<uint> value, uint candidate)
{
    RWByteAddressBuffer resource = pmGraphicsResource(value);
    uint original;
    resource.InterlockedMax(value.byte_offset, candidate, original);
    return original;
}
public uint pmAtomicMin(PMBuffer<uint> value, uint candidate)
{
    RWByteAddressBuffer resource = pmGraphicsResource(value);
    uint original;
    resource.InterlockedMin(value.byte_offset, candidate, original);
    return original;
}
public uint pmAtomicOr(PMBuffer<uint> value, uint candidate)
{
    RWByteAddressBuffer resource = pmGraphicsResource(value);
    uint original;
    resource.InterlockedOr(value.byte_offset, candidate, original);
    return original;
}
public uint64_t pmAtomicOr(PMBuffer<uint64_t> value, uint64_t candidate)
{
    RWByteAddressBuffer resource = pmGraphicsResource(value);
    return resource.InterlockedOrU64(
        value.byte_offset, candidate);
}
public uint pmAtomicExchange(PMBuffer<uint> value, uint replacement)
{
    RWByteAddressBuffer resource = pmGraphicsResource(value);
    uint original;
    resource.InterlockedExchange(
        value.byte_offset, replacement, original);
    return original;
}

public float pmAtomicAddRawFloat(
    RWByteAddressBuffer resource, uint address, float addend)
{
    uint expected = resource.Load<uint>(address);
    for (;;)
    {
        uint replacement = asuint(asfloat(expected) + addend);
        uint original;
        resource.InterlockedCompareExchange(
            address, expected, replacement, original);
        if (original == expected) return asfloat(original);
        expected = original;
    }
}

public void pmAtomicAddVec3(PMBuffer<Vec3> value, Vec3 addend)
{
    RWByteAddressBuffer resource = pmGraphicsResource(value);
    uint address = value.byte_offset;
    pmAtomicAddRawFloat(resource, address, addend.x);
    pmAtomicAddRawFloat(resource, address + sizeof(float), addend.y);
    pmAtomicAddRawFloat(resource, address + 2u * sizeof(float), addend.z);
}

public uint pmAtomicMaxPositiveFloat(PMBuffer<float> value, float candidate)
{
    RWByteAddressBuffer resource = pmGraphicsResource(value);
    uint original;
    resource.InterlockedMax(
        value.byte_offset, asuint(candidate), original);
    return original;
}

public uint pmAtomicMinPositiveFloat(PMBuffer<float> value, float candidate)
{
    RWByteAddressBuffer resource = pmGraphicsResource(value);
    uint original;
    resource.InterlockedMin(
        value.byte_offset, asuint(candidate), original);
    return original;
}
#else
public void pmAtomicAddVec3(PMBuffer<Vec3> value, Vec3 addend)
{
    pmAtomicAdd(&value[0].x, addend.x);
    pmAtomicAdd(&value[0].y, addend.y);
    pmAtomicAdd(&value[0].z, addend.z);
}
public uint pmAtomicMaxPositiveFloat(PMBuffer<float> value, float candidate)
{
    return pmAtomicMax(reinterpret<uint*>(value), asuint(candidate));
}
public uint pmAtomicMinPositiveFloat(PMBuffer<float> value, float candidate)
{
    return pmAtomicMin(reinterpret<uint*>(value), asuint(candidate));
}
#endif
public void pmSyncWarp(uint) { GroupMemoryBarrierWithGroupSync(); }
public uint3 pmGridDim() { return pmDispatchGroupCount; }

// Graphics APIs have no in-dispatch grid barrier. Their thin device adapters
// split logical schedules at each grid dependency; this remains a group
// barrier within one dispatch.
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

public void pmAtomicAddVec3(PMBuffer<Vec3> value, Vec3 addend)
{
    pmAtomicAdd(&value[0].x, addend.x);
    pmAtomicAdd(&value[0].y, addend.y);
    pmAtomicAdd(&value[0].z, addend.z);
}

public uint pmAtomicMaxPositiveFloat(PMBuffer<float> value, float candidate)
{
    return pmAtomicMax(reinterpret<uint*>(value), asuint(candidate));
}

public uint pmAtomicMinPositiveFloat(PMBuffer<float> value, float candidate)
{
    return pmAtomicMin(reinterpret<uint*>(value), asuint(candidate));
}
#endif

#if defined(PM_TARGET_VULKAN) || PM_TARGET_D3D12
public uint pmThreadIndex() { return pmDispatchThreadId.x; }
#endif

#if !defined(PM_TARGET_CPU)
public void pmGroupBarrier() { GroupMemoryBarrierWithGroupSync(); }
#endif
