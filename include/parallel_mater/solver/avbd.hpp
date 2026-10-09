// SPDX-License-Identifier: MIT
#pragma once

// Augmented Vertex Block Descent (Giles, Diaz, Yuksel, SIGGRAPH 2025).
// Shared numerical contract: host, CUDA and Metal compile this same source.
// Backend adapters own geometry, graph coloring, storage and synchronization.
#ifdef __METAL_VERSION__
#define PM_AVBD_INLINE inline
#define PM_AVBD_THREAD thread
#define PM_AVBD_CONSTANT constant constexpr
#else
#include <cmath>
#define PM_AVBD_THREAD
#define PM_AVBD_CONSTANT constexpr
#ifdef __CUDACC__
#define PM_AVBD_INLINE __host__ __device__ inline
#else
#define PM_AVBD_INLINE inline
#endif
#endif

namespace parallel_mater { namespace avbd {
PM_AVBD_CONSTANT float alpha = 0.99f;
PM_AVBD_CONSTANT float gamma = 0.999f;
PM_AVBD_CONSTANT float minimum_penalty = 1.0f;
PM_AVBD_CONSTANT float maximum_penalty = 1.0e10f;
PM_AVBD_CONSTANT float linear_beta = 10000.0f;
PM_AVBD_CONSTANT float angular_beta = 100.0f;
PM_AVBD_CONSTANT unsigned default_iterations = 10;

PM_AVBD_INLINE float min_value(float a, float b) { return a < b ? a : b; }
PM_AVBD_INLINE float max_value(float a, float b) { return a > b ? a : b; }
PM_AVBD_INLINE float abs_value(float a) { return a < 0 ? -a : a; }
PM_AVBD_INLINE float root(float a) {
#ifdef __METAL_VERSION__
    return metal::sqrt(a);
#else
    return sqrtf(a);
#endif
}
PM_AVBD_INLINE float clamp(float x, float lo, float hi) {
    return min_value(max_value(x, lo), hi);
}

// Multipliers are forces (linear rows) or torques (angular rows), not impulses.
// Convert with lambda * h at API edges.
struct Dual {
    float lambda{};
    float penalty{minimum_penalty};
};
PM_AVBD_INLINE Dual warm_start(Dual value, float stiffness = maximum_penalty) {
    value.lambda *= alpha * gamma;
    value.penalty = min_value(stiffness,
        clamp(value.penalty * gamma, minimum_penalty, maximum_penalty));
    return value;
}
PM_AVBD_INLINE float force(Dual value, float error,
                           float lower = -maximum_penalty,
                           float upper = maximum_penalty) {
    return clamp(value.lambda + value.penalty * error, lower, upper);
}
PM_AVBD_INLINE Dual update_dual(Dual value, float error, float beta,
                               float lower = -maximum_penalty,
                               float upper = maximum_penalty,
                               float stiffness = maximum_penalty) {
    const float unconstrained = value.lambda + value.penalty * error;
    value.lambda = stiffness < maximum_penalty ? 0.0f : clamp(unconstrained, lower, upper);
    if (unconstrained > lower && unconstrained < upper)
        value.penalty = min_value(stiffness, min_value(maximum_penalty,
            value.penalty + beta * abs_value(error)));
    return value;
}

struct ContactForce { float normal{}, tangent0{}, tangent1{}; };
PM_AVBD_INLINE ContactForce project_contact(ContactForce value, float friction) {
    value.normal = max_value(value.normal, 0.0f);
    const float bound = friction * value.normal;
    const float tangent_length = root(value.tangent0 * value.tangent0 + value.tangent1 * value.tangent1);
    if (tangent_length > bound && tangent_length > 0.0f) {
        value.tangent0 *= bound / tangent_length;
        value.tangent1 *= bound / tangent_length;
    }
    return value;
}

// PSD approximation of the clamped force derivative. Inactive constraints
// have no curvature; in particular friction=0 must not add artificial
// tangential stiffness to a body's otherwise free motion.
PM_AVBD_INLINE ContactForce contact_stiffness_scales(ContactForce trial, float friction) {
    const float normal = max_value(trial.normal, 0.0f);
    const float bound = friction * normal;
    const float tangent = root(trial.tangent0 * trial.tangent0 + trial.tangent1 * trial.tangent1);
    const float scale = bound <= 0 ? 0 : tangent <= bound ? 1 : bound / tangent;
    return {trial.normal > 0 ? 1.0f : 0.0f, scale, scale};
}

struct Vector6 { float v[6]{}; };
PM_AVBD_INLINE float dot6(Vector6 a, Vector6 b) {
    float result = 0;
    for (unsigned i = 0; i < 6; ++i) result += a.v[i] * b.v[i];
    return result;
}

// A full coupled translation/rotation block, not six independent scalar rows.
struct Block {
    float h[6][6]{};
    float g[6]{};
};
PM_AVBD_INLINE void add_row(PM_AVBD_THREAD Block &block, Vector6 jacobian,
                            float stiffness, float row_force) {
    for (unsigned i = 0; i < 6; ++i) {
        block.g[i] += jacobian.v[i] * row_force;
        for (unsigned j = 0; j <= i; ++j)
            block.h[i][j] += stiffness * jacobian.v[i] * jacobian.v[j];
    }
}

// LDL^T, using the lower triangle. Stiff rank-one rows can round a small
// inertial pivot out of a float matrix. Only when that cancellation occurs,
// equilibrate and retry with a roundoff-sized modified-Cholesky pivot floor.
// This repairs numerical curvature, not physical force limits. Well-resolved
// blocks avoid unnecessary scaling roundoff; invalid input still fails.
PM_AVBD_INLINE bool solve(Block input, PM_AVBD_THREAD Vector6 &delta) {
    constexpr float largest = 3.402823466e38f;
    constexpr float epsilon = 1.1920928955078125e-7f;
    constexpr float pivot_floor = 8.0f * epsilon;
    constexpr float roundoff_tolerance = 64.0f * epsilon;
    for (unsigned i = 0; i < 6; ++i) {
        if (!(input.h[i][i] > 0.0f) || !(input.h[i][i] < largest)
            || !(abs_value(input.g[i]) < largest)) return false;
        for (unsigned j = 0; j < i; ++j)
            if (!(abs_value(input.h[i][j]) < largest)) return false;
    }
    for (unsigned attempt = 0; attempt < 2; ++attempt) {
        Block block = input;
        float scale[6]{};
        float d[6]{};
        float y[6]{};
        for (unsigned i = 0; i < 6; ++i)
            scale[i] = attempt ? 1.0f / root(block.h[i][i]) : 1.0f;
        if (attempt) {
            for (unsigned i = 0; i < 6; ++i) {
                block.g[i] *= scale[i];
                if (!(abs_value(block.g[i]) < largest)) return false;
                for (unsigned j = 0; j < i; ++j) {
                    block.h[i][j] = block.h[i][j] * scale[i] * scale[j];
                    // Every PSD matrix has |H_ij| <= sqrt(H_ii H_jj).
                    if (!(abs_value(block.h[i][j]) <= 1.0f + roundoff_tolerance)) return false;
                }
                block.h[i][i] = 1.0f;
            }
        }
        bool retry = false;
        for (unsigned i = 0; i < 6; ++i) {
            for (unsigned j = 0; j < i; ++j) {
                float sum = block.h[i][j];
                for (unsigned k = 0; k < j; ++k)
                    sum -= block.h[i][k] * d[k] * block.h[j][k];
                block.h[i][j] = sum / d[j];
            }
            float diagonal = block.h[i][i];
            for (unsigned k = 0; k < i; ++k)
                diagonal -= block.h[i][k] * block.h[i][k] * d[k];
            if (!attempt) {
                if (!(diagonal > pivot_floor * input.h[i][i]) || !(diagonal < largest)) {
                    retry = true;
                    break;
                }
                d[i] = diagonal;
            } else {
                if (!(diagonal >= -roundoff_tolerance) || !(diagonal < largest)) return false;
                d[i] = max_value(diagonal, pivot_floor);
            }
            y[i] = -block.g[i];
            for (unsigned k = 0; k < i; ++k) y[i] -= block.h[i][k] * y[k];
        }
        if (retry) continue;
        for (int i = 5; i >= 0; --i) {
            delta.v[i] = y[i] / d[i];
            for (unsigned k = unsigned(i) + 1; k < 6; ++k)
                delta.v[i] -= block.h[k][i] * delta.v[k];
        }
        for (unsigned i = 0; i < 6; ++i) {
            delta.v[i] *= scale[i];
            if (!(abs_value(delta.v[i]) < largest)) return false;
        }
        return true;
    }
    return false;
}

// Linearized bilateral/limit/spring/motor row. All adapters use these exact
// primal and dual equations; a joint is just a small collection of rows.
struct Row {
    Vector6 a{}, b{};
    Dual dual{};
    float error{}, velocity{};
    float lower{-maximum_penalty}, upper{maximum_penalty};
    float stiffness{maximum_penalty}, damping{};
    float beta{linear_beta};
    // A corrective AVBD phase may retain total dual forces while minimizing
    // increments relative to forces already applied by the primary phase.
    float reference_force{};
};
PM_AVBD_INLINE float row_error(Row row, Vector6 a, Vector6 b) {
    return row.error + dot6(row.a, a) + dot6(row.b, b);
}
PM_AVBD_INLINE float row_trial(Row row, Vector6 a, Vector6 b, float dt) {
    const float displacement = dot6(row.a, a) + dot6(row.b, b);
    return row.dual.lambda + row.dual.penalty * (row.error + displacement)
        + row.damping * (row.velocity + displacement / dt);
}
PM_AVBD_INLINE float row_force(Row row, Vector6 a, Vector6 b, float dt) {
    return clamp(row_trial(row, a, b, dt), row.lower, row.upper);
}
PM_AVBD_INLINE void accumulate(PM_AVBD_THREAD Block &block, Row row,
                               bool first, Vector6 a, Vector6 b, float dt) {
    const float trial = row_trial(row, a, b, dt);
    // Dormant one-sided limits add no curvature: their historical penalty
    // must not artificially resist motion within the permitted interval.
    // For two-sided force caps (notably motors), retain the PSD majorant.
    // An unsafeguarded semismooth Newton step with zero saturated curvature
    // can jump across BOTH caps, alternate between them, and inject energy.
    // All stored bounds are finite, including the large numerical hard cap.
    const bool two_sided = row.lower < 0.0f && row.upper > 0.0f;
    const float stiffness = two_sided || (trial > row.lower && trial < row.upper)
        ? row.dual.penalty + row.damping / dt : 0.0f;
    add_row(block, first ? row.a : row.b,
        stiffness, clamp(trial, row.lower, row.upper) - row.reference_force);
}
PM_AVBD_INLINE Row advance(Row row, Vector6 a, Vector6 b) {
    row.dual = update_dual(row.dual, row_error(row, a, b), row.beta,
        row.lower, row.upper, row.stiffness);
    return row;
}
} }
#undef PM_AVBD_INLINE
#undef PM_AVBD_THREAD

#undef PM_AVBD_CONSTANT
