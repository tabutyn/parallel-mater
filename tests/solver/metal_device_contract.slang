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

struct MetalRecordContract
{
    bool enabled;
    bool negate;
    Vec3 gain;
    float* inputs[2];
    float* output;
    uint count;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void metal_record_contract(
    uniform MetalRecordContract inline_record,
    uniform MetalRecordContract* records)
{
    uint index = pmThreadIndex();
    if (index < inline_record.count && inline_record.enabled)
        inline_record.output[index] = (inline_record.negate ? -1.0f : 1.0f) *
            (inline_record.inputs[0][index] * inline_record.gain.x +
             inline_record.inputs[1][index] * inline_record.gain.y);
    if (index < records[0].count && records[0].enabled)
        records[0].output[index] = (records[0].negate ? -1.0f : 1.0f) *
            (records[0].inputs[0][index] * records[0].gain.x +
             records[0].inputs[1][index] * records[0].gain.y);
}
