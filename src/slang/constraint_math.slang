// SPDX-License-Identifier: MIT
// AVBD block, dual, and row math shared by the runtime solver and the
// cross-target conformance entry point.

#ifndef PM_GEOMETRY_CONSTRAINTS_ADAPTER
#define PM_GEOMETRY_CONSTRAINTS_ADAPTER 0
#endif
#ifndef PM_AVBD_SCHUR_MODULE
#define PM_AVBD_SCHUR_MODULE 0
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

float avbdFrictionScale(float tangent0, float tangent1, float bound)
{
    if (bound <= 0.0f) return 0.0f;
    float squared = tangent0 * tangent0 + tangent1 * tangent1;
    // One reciprocal square root avoids separately rounding sqrt and divide.
    // Projection and curvature must use the same radial Coulomb scale.
    return squared <= bound * bound ? 1.0f
        : avbdMin(1.0f, bound * rsqrt(squared));
}

AvbdContactForce avbdProjectContact(AvbdContactForce value, float friction)
{
    value.normal = avbdMax(value.normal, 0.0f);
    float bound = friction * value.normal;
    float scale = avbdFrictionScale(value.tangent0, value.tangent1, bound);
    value.tangent0 *= scale;
    value.tangent1 *= scale;
    return value;
}

AvbdContactForce avbdContactStiffnessScales(
    AvbdContactForce trial, float friction)
{
    float normal = avbdMax(trial.normal, 0.0f);
    float bound = friction * normal;
    float scale = avbdFrictionScale(trial.tangent0, trial.tangent1, bound);
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
    // Symmetric lower triangle: row * (row + 1) / 2 + column.
    // The solver never consumes the redundant upper triangle; packing it
    // reduces mobile shader private storage from 42 to 27 floats per block.
    float h[21];
    float g[6];
};

uint avbdBlockIndex(uint row, uint column)
{
    uint high = max(row, column);
    uint low = min(row, column);
    return high * (high + 1u) / 2u + low;
}

AvbdBlock avbdZeroBlock()
{
    AvbdBlock result;
    [ForceUnroll]
    for (uint i = 0; i < 6; ++i) result.g[i] = 0.0f;
    [ForceUnroll]
    for (uint i = 0; i < 21; ++i) result.h[i] = 0.0f;
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
            block.h[avbdBlockIndex(i, j)] += stiffness
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
        if (!(input.h[avbdBlockIndex(i, i)] > 0.0f) ||
            !(input.h[avbdBlockIndex(i, i)] < largest) ||
            !(avbdAbs(input.g[i]) < largest)) return false;
        [loop]
        for (uint j = 0; j < i; ++j)
            if (!(avbdAbs(input.h[avbdBlockIndex(i, j)]) < largest))
                return false;
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
            scale[i] = attempt != 0
                ? rsqrt(block.h[avbdBlockIndex(i, i)]) : 1.0f;
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
                    block.h[avbdBlockIndex(i, j)] *= scale[i] * scale[j];
                    if (!(avbdAbs(block.h[avbdBlockIndex(i, j)])
                          <= 1.0f + roundoffTolerance)) return false;
                }
                block.h[avbdBlockIndex(i, i)] = 1.0f;
            }
        }
        bool retry = false;
        [ForceUnroll]
        for (uint i = 0; i < 6; ++i)
        {
            [loop]
            for (uint j = 0; j < i; ++j)
            {
                float sum = block.h[avbdBlockIndex(i, j)];
                [loop]
                for (uint k = 0; k < j; ++k)
                    sum -= block.h[avbdBlockIndex(i, k)] * diagonalFactors[k]
                         * block.h[avbdBlockIndex(j, k)];
                block.h[avbdBlockIndex(i, j)] = sum / diagonalFactors[j];
            }
            float diagonal = block.h[avbdBlockIndex(i, i)];
            [loop]
            for (uint k = 0; k < i; ++k)
                diagonal -= block.h[avbdBlockIndex(i, k)] *
                            block.h[avbdBlockIndex(i, k)]
                          * diagonalFactors[k];
            if (attempt == 0)
            {
                if (!(diagonal > pivotFloor *
                        input.h[avbdBlockIndex(i, i)])
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
                forward[i] -= block.h[avbdBlockIndex(i, k)] * forward[k];
        }
        if (retry) continue;
        [ForceUnroll]
        for (int i = 5; i >= 0; --i)
        {
            delta.values[i] = forward[i] / diagonalFactors[i];
            [loop]
            for (uint k = uint(i) + 1; k < 6; ++k)
                delta.values[i] -=
                    block.h[avbdBlockIndex(k, uint(i))] * delta.values[k];
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

#if PM_AVBD_SCHUR_MODULE
struct AvbdSymmetric3
{
    float xx;
    float yx;
    float yy;
    float zx;
    float zy;
    float zz;
};

struct AvbdVector3
{
    float x;
    float y;
    float z;
};

bool avbdInverseSymmetric3(AvbdSymmetric3 value, out AvbdSymmetric3 inverse)
{
    const float largest = 3.402823466e38f;
    const float pivotFloor = 32.0f * 1.1920928955078125e-7f;
    inverse = AvbdSymmetric3(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    if (!(value.xx > 0.0f) || !(value.yy > 0.0f) ||
        !(value.zz > 0.0f) || !(value.xx < largest) ||
        !(value.yy < largest) || !(value.zz < largest))
        return false;
    float leadingMinor = value.xx * value.yy - value.yx * value.yx;
    if (!(leadingMinor > pivotFloor * value.xx * value.yy) ||
        !(leadingMinor < largest))
        return false;
    inverse.xx = value.yy * value.zz - value.zy * value.zy;
    inverse.yx = value.zx * value.zy - value.yx * value.zz;
    inverse.yy = value.xx * value.zz - value.zx * value.zx;
    inverse.zx = value.yx * value.zy - value.zx * value.yy;
    inverse.zy = value.yx * value.zx - value.xx * value.zy;
    inverse.zz = value.xx * value.yy - value.yx * value.yx;
    float determinant = value.xx * inverse.xx + value.yx * inverse.yx +
        value.zx * inverse.zx;
    if (!(determinant > pivotFloor * leadingMinor * value.zz) ||
        !(determinant < largest))
        return false;
    float reciprocal = 1.0f / determinant;
    inverse.xx *= reciprocal;
    inverse.yx *= reciprocal;
    inverse.yy *= reciprocal;
    inverse.zx *= reciprocal;
    inverse.zy *= reciprocal;
    inverse.zz *= reciprocal;
    return avbdAbs(inverse.xx) < largest &&
        avbdAbs(inverse.yx) < largest && avbdAbs(inverse.yy) < largest &&
        avbdAbs(inverse.zx) < largest && avbdAbs(inverse.zy) < largest &&
        avbdAbs(inverse.zz) < largest;
}

AvbdVector3 avbdMultiplySymmetric3(AvbdSymmetric3 value, AvbdVector3 vector)
{
    return AvbdVector3(
        value.xx * vector.x + value.yx * vector.y + value.zx * vector.z,
        value.yx * vector.x + value.yy * vector.y + value.zy * vector.z,
        value.zx * vector.x + value.zy * vector.y + value.zz * vector.z);
}

float avbdDot3(AvbdVector3 a, AvbdVector3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Exact block elimination of the same symmetric 6x6 system. A scaled
// backward-error check retains the established LDLT path for numerically
// awkward matrices.
bool avbdSolveSchur(AvbdBlock input, out AvbdVector6 delta)
{
    const float largest = 3.402823466e38f;
    const float epsilon = 1.1920928955078125e-7f;
    [loop]
    for (uint i = 0u; i < 6u; ++i)
        if (!(input.h[avbdBlockIndex(i, i)] > 0.0f) ||
            !(input.h[avbdBlockIndex(i, i)] < largest) ||
            !(avbdAbs(input.g[i]) < largest))
            return avbdSolve(input, delta);
    [loop]
    for (uint i = 0u; i < 21u; ++i)
        if (!(avbdAbs(input.h[i]) < largest))
            return avbdSolve(input, delta);

    AvbdSymmetric3 inverseA;
    if (!avbdInverseSymmetric3(AvbdSymmetric3(
            input.h[0], input.h[1], input.h[2],
            input.h[3], input.h[4], input.h[5]), inverseA))
        return avbdSolve(input, delta);
    AvbdVector3 b0 = AvbdVector3(input.h[6], input.h[7], input.h[8]);
    AvbdVector3 b1 = AvbdVector3(input.h[10], input.h[11], input.h[12]);
    AvbdVector3 b2 = AvbdVector3(input.h[15], input.h[16], input.h[17]);
    AvbdVector3 q0 = avbdMultiplySymmetric3(inverseA, b0);
    AvbdVector3 q1 = avbdMultiplySymmetric3(inverseA, b1);
    AvbdVector3 q2 = avbdMultiplySymmetric3(inverseA, b2);
    AvbdSymmetric3 inverseS;
    if (!avbdInverseSymmetric3(AvbdSymmetric3(
            input.h[9] - avbdDot3(b0, q0),
            input.h[13] - avbdDot3(b1, q0),
            input.h[14] - avbdDot3(b1, q1),
            input.h[18] - avbdDot3(b2, q0),
            input.h[19] - avbdDot3(b2, q1),
            input.h[20] - avbdDot3(b2, q2)), inverseS))
        return avbdSolve(input, delta);

    AvbdVector3 upper = avbdMultiplySymmetric3(inverseA,
        AvbdVector3(-input.g[0], -input.g[1], -input.g[2]));
    AvbdVector3 lower = avbdMultiplySymmetric3(inverseS, AvbdVector3(
        -input.g[3] - avbdDot3(b0, upper),
        -input.g[4] - avbdDot3(b1, upper),
        -input.g[5] - avbdDot3(b2, upper)));
    upper.x -= q0.x * lower.x + q1.x * lower.y + q2.x * lower.z;
    upper.y -= q0.y * lower.x + q1.y * lower.y + q2.y * lower.z;
    upper.z -= q0.z * lower.x + q1.z * lower.y + q2.z * lower.z;
    delta.values[0] = upper.x;
    delta.values[1] = upper.y;
    delta.values[2] = upper.z;
    delta.values[3] = lower.x;
    delta.values[4] = lower.y;
    delta.values[5] = lower.z;

    float maximumResidual = 0.0f;
    float maximumScale = 1.0f;
    [loop]
    for (uint row = 0u; row < 6u; ++row)
    {
        float residual = input.g[row];
        float scale = avbdAbs(input.g[row]);
        [loop]
        for (uint column = 0u; column < 6u; ++column)
        {
            float term = input.h[avbdBlockIndex(row, column)] *
                delta.values[column];
            residual += term;
            scale += avbdAbs(term);
        }
        maximumResidual = avbdMax(maximumResidual, avbdAbs(residual));
        maximumScale = avbdMax(maximumScale, scale);
    }
    if (!(maximumResidual <= 256.0f * epsilon * maximumScale))
        return avbdSolve(input, delta);
    return true;
}
#endif

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

#if !PM_GEOMETRY_CONSTRAINTS_ADAPTER
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
            [loop]
            for (uint j = 0; j <= i; ++j)
                block.h[avbdBlockIndex(i, j)] =
                    input.values[i * 6 + j];
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
        for (uint i = 0; i < 6; ++i)
            block.h[avbdBlockIndex(i, i)] = input.values[35 + i];
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
            diagonalSum += block.h[avbdBlockIndex(i, i)];
        }
        output.values[5] = gradientSum;
        output.values[6] = diagonalSum;
    }
    avbdOutputs[threadId.x] = output;
}
#endif
