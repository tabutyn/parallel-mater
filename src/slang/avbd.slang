// SPDX-License-Identifier: MIT
// Portable AVBD numerical core. The surrounding geometry, graph construction,
// storage, and dispatch adapters remain backend-specific.

#ifndef PM_AVBD_RUNTIME
#define PM_AVBD_RUNTIME 0
#endif

static const float kAvbdAlpha = 0.99f;
static const float kAvbdGamma = 0.999f;
static const float kAvbdMinimumPenalty = 1.0f;
static const float kAvbdMaximumPenalty = 1.0e10f;
static const float kAvbdLinearBeta = 10000.0f;
static const float kAvbdAngularBeta = 100.0f;
static const uint kAvbdDefaultIterations = 10;

float avbdMin(float a, float b) { return a < b ? a : b; }
float avbdMax(float a, float b) { return a > b ? a : b; }
float avbdAbs(float value) { return value < 0.0f ? -value : value; }
float avbdClamp(float value, float lower, float upper)
{
    return avbdMin(avbdMax(value, lower), upper);
}

struct AvbdDual
{
    float lambda;
    float penalty;
};

AvbdDual avbdWarmStart(AvbdDual value, float stiffness)
{
    value.lambda *= kAvbdAlpha * kAvbdGamma;
    value.penalty = avbdMin(stiffness,
        avbdClamp(value.penalty * kAvbdGamma,
                  kAvbdMinimumPenalty, kAvbdMaximumPenalty));
    return value;
}

float avbdForce(AvbdDual value, float error, float lower, float upper)
{
    return avbdClamp(value.lambda + value.penalty * error, lower, upper);
}

AvbdDual avbdUpdateDual(AvbdDual value, float error, float beta,
                        float lower, float upper, float stiffness)
{
    float unconstrained = value.lambda + value.penalty * error;
    value.lambda = stiffness < kAvbdMaximumPenalty
        ? 0.0f : avbdClamp(unconstrained, lower, upper);
    if (unconstrained > lower && unconstrained < upper)
    {
        value.penalty = avbdMin(stiffness, avbdMin(kAvbdMaximumPenalty,
            value.penalty + beta * avbdAbs(error)));
    }
    return value;
}

struct AvbdContactForce
{
    float normal;
    float tangent0;
    float tangent1;
};

AvbdContactForce avbdProjectContact(AvbdContactForce value, float friction)
{
    value.normal = avbdMax(value.normal, 0.0f);
    float bound = friction * value.normal;
    float tangentLength = sqrt(value.tangent0 * value.tangent0
                             + value.tangent1 * value.tangent1);
    if (tangentLength > bound && tangentLength > 0.0f)
    {
        value.tangent0 *= bound / tangentLength;
        value.tangent1 *= bound / tangentLength;
    }
    return value;
}

AvbdContactForce avbdContactStiffnessScales(
    AvbdContactForce trial, float friction)
{
    float normal = avbdMax(trial.normal, 0.0f);
    float bound = friction * normal;
    float tangent = sqrt(trial.tangent0 * trial.tangent0
                       + trial.tangent1 * trial.tangent1);
    float scale = bound <= 0.0f ? 0.0f
                : tangent <= bound ? 1.0f : bound / tangent;
    AvbdContactForce result;
    result.normal = trial.normal > 0.0f ? 1.0f : 0.0f;
    result.tangent0 = scale;
    result.tangent1 = scale;
    return result;
}

struct AvbdVector6
{
    float values[6];
};

float avbdVectorElement(AvbdVector6 value, uint index)
{
    return value.values[index];
}

AvbdVector6 avbdZeroVector6()
{
    AvbdVector6 result;
    [ForceUnroll]
    for (uint i = 0; i < 6; ++i) result.values[i] = 0.0f;
    return result;
}

float avbdDot6(AvbdVector6 a, AvbdVector6 b)
{
    float result = 0.0f;
    [ForceUnroll]
    for (uint i = 0; i < 6; ++i)
        result += avbdVectorElement(a, i) * avbdVectorElement(b, i);
    return result;
}

struct AvbdBlock
{
    float h[6][6];
    float g[6];
};

AvbdBlock avbdZeroBlock()
{
    AvbdBlock result;
    [ForceUnroll]
    for (uint i = 0; i < 6; ++i)
    {
        result.g[i] = 0.0f;
        [ForceUnroll]
        for (uint j = 0; j < 6; ++j) result.h[i][j] = 0.0f;
    }
    return result;
}

void avbdAddRow(inout AvbdBlock block, AvbdVector6 jacobian,
                float stiffness, float rowForce)
{
    [ForceUnroll]
    for (uint i = 0; i < 6; ++i)
    {
        block.g[i] += jacobian.values[i] * rowForce;
        [loop]
        for (uint j = 0; j <= i; ++j)
            block.h[i][j] += stiffness
                * jacobian.values[i] * jacobian.values[j];
    }
}

bool avbdSolve(AvbdBlock input, out AvbdVector6 delta)
{
    const float largest = 3.402823466e38f;
    const float epsilon = 1.1920928955078125e-7f;
    const float pivotFloor = 8.0f * epsilon;
    const float roundoffTolerance = 64.0f * epsilon;
    delta = avbdZeroVector6();
    [ForceUnroll]
    for (uint i = 0; i < 6; ++i)
    {
        if (!(input.h[i][i] > 0.0f) || !(input.h[i][i] < largest)
            || !(avbdAbs(input.g[i]) < largest)) return false;
        [loop]
        for (uint j = 0; j < i; ++j)
            if (!(avbdAbs(input.h[i][j]) < largest)) return false;
    }
    [ForceUnroll]
    for (uint attempt = 0; attempt < 2; ++attempt)
    {
        AvbdBlock block = input;
        float scale[6];
        float diagonalFactors[6];
        float forward[6];
        [ForceUnroll]
        for (uint i = 0; i < 6; ++i)
        {
            scale[i] = attempt != 0 ? 1.0f / sqrt(block.h[i][i]) : 1.0f;
            diagonalFactors[i] = 0.0f;
            forward[i] = 0.0f;
        }
        if (attempt != 0)
        {
            [ForceUnroll]
            for (uint i = 0; i < 6; ++i)
            {
                block.g[i] *= scale[i];
                if (!(avbdAbs(block.g[i]) < largest)) return false;
                [loop]
                for (uint j = 0; j < i; ++j)
                {
                    block.h[i][j] *= scale[i] * scale[j];
                    if (!(avbdAbs(block.h[i][j])
                          <= 1.0f + roundoffTolerance)) return false;
                }
                block.h[i][i] = 1.0f;
            }
        }
        bool retry = false;
        [ForceUnroll]
        for (uint i = 0; i < 6; ++i)
        {
            [loop]
            for (uint j = 0; j < i; ++j)
            {
                float sum = block.h[i][j];
                [loop]
                for (uint k = 0; k < j; ++k)
                    sum -= block.h[i][k] * diagonalFactors[k]
                         * block.h[j][k];
                block.h[i][j] = sum / diagonalFactors[j];
            }
            float diagonal = block.h[i][i];
            [loop]
            for (uint k = 0; k < i; ++k)
                diagonal -= block.h[i][k] * block.h[i][k]
                          * diagonalFactors[k];
            if (attempt == 0)
            {
                if (!(diagonal > pivotFloor * input.h[i][i])
                    || !(diagonal < largest))
                {
                    retry = true;
                    break;
                }
                diagonalFactors[i] = diagonal;
            }
            else
            {
                if (!(diagonal >= -roundoffTolerance)
                    || !(diagonal < largest)) return false;
                diagonalFactors[i] = avbdMax(diagonal, pivotFloor);
            }
            forward[i] = -block.g[i];
            [loop]
            for (uint k = 0; k < i; ++k)
                forward[i] -= block.h[i][k] * forward[k];
        }
        if (retry) continue;
        [ForceUnroll]
        for (int i = 5; i >= 0; --i)
        {
            delta.values[i] = forward[i] / diagonalFactors[i];
            [loop]
            for (uint k = uint(i) + 1; k < 6; ++k)
                delta.values[i] -= block.h[k][i] * delta.values[k];
        }
        [ForceUnroll]
        for (uint i = 0; i < 6; ++i)
        {
            delta.values[i] *= scale[i];
            if (!(avbdAbs(delta.values[i]) < largest)) return false;
        }
        return true;
    }
    return false;
}

struct AvbdRow
{
    AvbdVector6 a;
    AvbdVector6 b;
    AvbdDual dual;
    float error;
    float velocity;
    float lower;
    float upper;
    float stiffness;
    float damping;
    float beta;
    float referenceForce;
};

float avbdRowVectorElement(AvbdRow row, uint side, uint index)
{
    return avbdVectorElement(side == 0u ? row.a : row.b, index);
}

float avbdRowScalarElement(AvbdRow row, uint member)
{
    if (member == 0u) return row.dual.lambda;
    if (member == 1u) return row.dual.penalty;
    if (member == 2u) return row.error;
    if (member == 3u) return row.velocity;
    if (member == 4u) return row.lower;
    if (member == 5u) return row.upper;
    if (member == 6u) return row.stiffness;
    if (member == 7u) return row.damping;
    if (member == 8u) return row.beta;
    return row.referenceForce;
}

float avbdRowError(AvbdRow row, AvbdVector6 a, AvbdVector6 b)
{
    return row.error + avbdDot6(row.a, a) + avbdDot6(row.b, b);
}

float avbdRowTrial(AvbdRow row, AvbdVector6 a, AvbdVector6 b, float dt)
{
    float displacement = avbdDot6(row.a, a) + avbdDot6(row.b, b);
    return row.dual.lambda + row.dual.penalty * (row.error + displacement)
        + row.damping * (row.velocity + displacement / dt);
}

float avbdRowForce(AvbdRow row, AvbdVector6 a, AvbdVector6 b, float dt)
{
    return avbdClamp(avbdRowTrial(row, a, b, dt), row.lower, row.upper);
}

void avbdAccumulate(inout AvbdBlock block, AvbdRow row, bool first,
                    AvbdVector6 a, AvbdVector6 b, float dt)
{
    float trial = avbdRowTrial(row, a, b, dt);
    bool twoSided = row.lower < 0.0f && row.upper > 0.0f;
    float stiffness = twoSided || (trial > row.lower && trial < row.upper)
        ? row.dual.penalty + row.damping / dt : 0.0f;
    avbdAddRow(block, first ? row.a : row.b, stiffness,
        avbdClamp(trial, row.lower, row.upper) - row.referenceForce);
}

AvbdRow avbdAdvance(AvbdRow row, AvbdVector6 a, AvbdVector6 b)
{
    row.dual = avbdUpdateDual(row.dual, avbdRowError(row, a, b), row.beta,
        row.lower, row.upper, row.stiffness);
    return row;
}

#if PM_AVBD_RUNTIME
// Keep every numerical primitive visible to production adapters without
// pulling the conformance resources and their global parameter block into the
// generated CUDA translation unit. This entry is compiled but never launched;
// the backend calls the generated device functions directly.
[shader("compute")]
[numthreads(1, 1, 1)]
void avbdRuntimeMain(
    uint3 threadId : SV_DispatchThreadID,
    uniform AvbdBlock* blocks,
    uniform AvbdRow* rows,
    uniform AvbdVector6* vectors,
    uniform AvbdDual* duals,
    uniform AvbdContactForce* contactForces,
    uniform uint count)
{
    uint index = threadId.x;
    if (index >= count) return;
    uint vectorIndex = index * 2u;
    AvbdBlock block = blocks[index];
    AvbdRow row = rows[index];
    AvbdVector6 a = vectors[vectorIndex];
    AvbdVector6 b = vectors[vectorIndex + 1u];
    AvbdDual dual = avbdWarmStart(duals[index], row.stiffness);
    dual = avbdUpdateDual(dual, row.error, row.beta,
        row.lower, row.upper, row.stiffness);
    row.dual = dual;
    row.referenceForce += avbdForce(
        dual, row.error, row.lower, row.upper);
    AvbdContactForce force = avbdProjectContact(
        contactForces[index], row.damping);
    AvbdContactForce scales = avbdContactStiffnessScales(
        force, row.damping);
    row.velocity += avbdDot6(a, b) + avbdVectorElement(a, 0u)
        + scales.normal;
    avbdAddRow(block, row.a, row.stiffness, row.referenceForce);
    avbdAccumulate(block, row, true, a, b, 1.0f);
    row.error = avbdRowError(row, a, b);
    row.velocity += avbdRowTrial(row, a, b, 1.0f);
    row.referenceForce += avbdRowForce(row, a, b, 1.0f);
    row = avbdAdvance(row, a, b);
    row.referenceForce += avbdRowVectorElement(row, 0u, 0u)
        + avbdRowScalarElement(row, 0u);
    AvbdVector6 update = avbdZeroVector6();
    if (avbdSolve(block, update)) vectors[vectorIndex] = update;
    blocks[index] = block;
    rows[index] = row;
    duals[index] = dual;
    contactForces[index] = force;
}
#else
// Stable cross-target ABI used by the conformance runner.
struct SlangAvbdInput
{
    uint operation;
    float values[64];
};

struct SlangAvbdOutput
{
    float values[8];
    uint solved;
};

StructuredBuffer<SlangAvbdInput> avbdInputs;
RWStructuredBuffer<SlangAvbdOutput> avbdOutputs;
uniform uint avbdInputCount;

AvbdVector6 avbdLoadVector(SlangAvbdInput input, uint offset)
{
    AvbdVector6 result;
    [ForceUnroll]
    for (uint i = 0; i < 6; ++i) result.values[i] = input.values[offset + i];
    return result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void avbdConformanceMain(uint3 threadId : SV_DispatchThreadID)
{
    if (threadId.x >= avbdInputCount) return;
    SlangAvbdInput input = avbdInputs[threadId.x];
    SlangAvbdOutput output;
    [ForceUnroll]
    for (uint i = 0; i < 8; ++i) output.values[i] = 0.0f;
    output.solved = 1;

    if (input.operation == 0)
    {
        AvbdBlock block = avbdZeroBlock();
        [ForceUnroll]
        for (uint i = 0; i < 6; ++i)
        {
            block.g[i] = input.values[36 + i];
            [ForceUnroll]
            for (uint j = 0; j < 6; ++j)
                block.h[i][j] = input.values[i * 6 + j];
        }
        AvbdVector6 delta;
        output.solved = avbdSolve(block, delta) ? 1 : 0;
        [ForceUnroll]
        for (uint i = 0; i < 6; ++i) output.values[i] = delta.values[i];
    }
    else if (input.operation == 1)
    {
        AvbdContactForce force;
        force.normal = input.values[0];
        force.tangent0 = input.values[1];
        force.tangent1 = input.values[2];
        AvbdContactForce projected = avbdProjectContact(force, input.values[3]);
        AvbdContactForce scales = avbdContactStiffnessScales(force, input.values[3]);
        output.values[0] = projected.normal;
        output.values[1] = projected.tangent0;
        output.values[2] = projected.tangent1;
        output.values[3] = scales.normal;
        output.values[4] = scales.tangent0;
        output.values[5] = scales.tangent1;
    }
    else if (input.operation == 2)
    {
        AvbdDual dual;
        dual.lambda = input.values[0];
        dual.penalty = input.values[1];
        AvbdDual warm = avbdWarmStart(dual, input.values[6]);
        AvbdDual advanced = avbdUpdateDual(dual, input.values[2], input.values[3],
            input.values[4], input.values[5], input.values[6]);
        output.values[0] = warm.lambda;
        output.values[1] = warm.penalty;
        output.values[2] = avbdForce(dual, input.values[2],
                                     input.values[4], input.values[5]);
        output.values[3] = advanced.lambda;
        output.values[4] = advanced.penalty;
    }
    else
    {
        AvbdRow row;
        row.a = avbdLoadVector(input, 0);
        row.b = avbdLoadVector(input, 6);
        row.dual.lambda = input.values[12];
        row.dual.penalty = input.values[13];
        row.error = input.values[14];
        row.velocity = input.values[15];
        row.lower = input.values[16];
        row.upper = input.values[17];
        row.stiffness = input.values[18];
        row.damping = input.values[19];
        row.beta = input.values[20];
        row.referenceForce = input.values[21];
        AvbdVector6 a = avbdLoadVector(input, 22);
        AvbdVector6 b = avbdLoadVector(input, 28);
        float dt = input.values[34];
        AvbdBlock block = avbdZeroBlock();
        [ForceUnroll]
        for (uint i = 0; i < 6; ++i) block.h[i][i] = input.values[35 + i];
        avbdAccumulate(block, row, input.values[41] != 0.0f, a, b, dt);
        AvbdRow advanced = avbdAdvance(row, a, b);
        output.values[0] = avbdRowError(row, a, b);
        output.values[1] = avbdRowTrial(row, a, b, dt);
        output.values[2] = avbdRowForce(row, a, b, dt);
        output.values[3] = avbdRowScalarElement(advanced, 0u);
        output.values[4] = avbdRowScalarElement(advanced, 1u);
        float gradientSum = 0.0f;
        float diagonalSum = 0.0f;
        [ForceUnroll]
        for (uint i = 0; i < 6; ++i)
        {
            gradientSum += block.g[i];
            diagonalSum += block.h[i][i];
        }
        output.values[5] = gradientSum;
        output.values[6] = diagonalSum;
    }
    avbdOutputs[threadId.x] = output;
}
#endif
