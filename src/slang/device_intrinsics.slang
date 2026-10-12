// SPDX-License-Identifier: MIT
// Portable dispatch and atomic operations used by shared physics kernels.

import types;

uint pmThreadIndex()
{
    return cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
}

__target_intrinsic(cuda, "make_uint3(gridDim.x, gridDim.y, gridDim.z)")
uint3 pmGridDim();

__target_intrinsic(cuda, "atomicAdd") uint pmAtomicAdd(uint* value, uint addend);
__target_intrinsic(cuda, "atomicAdd") uint64_t pmAtomicAdd(
    uint64_t* value, uint64_t addend);
__target_intrinsic(cuda, "atomicAdd") float pmAtomicAdd(float* value, float addend);
__target_intrinsic(cuda, "atomicMax") uint pmAtomicMax(uint* value, uint candidate);
__target_intrinsic(cuda, "atomicMin") uint pmAtomicMin(uint* value, uint candidate);
__target_intrinsic(cuda, "atomicOr") uint pmAtomicOr(uint* value, uint candidate);
__target_intrinsic(cuda, "atomicOr") uint64_t pmAtomicOr(
    uint64_t* value, uint64_t candidate);
__target_intrinsic(cuda, "atomicExch") uint pmAtomicExchange(uint* value, uint replacement);
__target_intrinsic(cuda, "__syncwarp") void pmSyncWarp(uint mask);
__target_intrinsic(cuda, "pm_cuda_grid_barrier()")
void pmNativeGridBarrier();

void pmGridBarrier()
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

void pmGroupBarrier() { GroupMemoryBarrierWithGroupSync(); }
