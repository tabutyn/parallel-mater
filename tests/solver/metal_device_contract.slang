// SPDX-License-Identifier: MIT
// Exercise adapter operations, not a second implementation of physics.
import types;
import device_intrinsics;

[shader("compute")]
[numthreads(128, 1, 1)]
void metal_device_contract(
    uniform uint count, uniform uint64_t addend,
    uniform uint64_t* total, uniform uint64_t* mask,
    uniform Vec3* vector_sum, uniform uint* indices,
    uniform float* maximum, uniform float* minimum)
{
    uint index = pmThreadIndex();
    if (index >= count) return;
    indices[index] = pmBlockIndex().x * pmGroupSize().x +
        pmGroupThreadIndex().x + pmGridDim().x * pmGroupSize().x;
    pmAccumulateU64(total, addend);
    pmAccumulateMaskU64(mask, 1ull << (index & 63u));
    Vec3 contribution = {1.0f, 2.0f, -1.0f};
    pmAtomicAddVec3(vector_sum, contribution);
    pmAtomicMaxPositiveFloat(maximum, float(index + 1u));
    pmAtomicMinPositiveFloat(minimum, float(index + 1u));
}
