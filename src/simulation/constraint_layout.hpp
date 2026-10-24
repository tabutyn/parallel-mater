// SPDX-License-Identifier: MIT
#pragma once

// Backend-neutral host storage mirrored by src/slang/constraint_math.slang.
// This header contains layouts and defaults only. Constraint equations and
// solver operations belong to Slang; CPU/CUDA reference implementations live
// with the conformance tests.

namespace parallel_mater::avbd {

#if defined(__CUDACC__)
#define PM_AVBD_LAYOUT_INLINE __host__ __device__ constexpr
#else
#define PM_AVBD_LAYOUT_INLINE constexpr
#endif

inline constexpr float alpha = 0.99F;
inline constexpr float gamma = 0.999F;
inline constexpr float minimum_penalty = 1.0F;
inline constexpr float maximum_penalty = 1.0e10F;
inline constexpr float linear_beta = 10000.0F;
inline constexpr float angular_beta = 100.0F;
inline constexpr unsigned default_iterations = 10U;

struct Dual {
    float lambda{};
    float penalty{minimum_penalty};
};

struct ContactForce {
    float normal{};
    float tangent0{};
    float tangent1{};
};

struct Vector6 {
    float v[6]{};
};

struct Symmetric6 {
    float values[21]{};

    PM_AVBD_LAYOUT_INLINE static unsigned index(
        unsigned row, unsigned column) noexcept {
        const unsigned high = row > column ? row : column;
        const unsigned low = row > column ? column : row;
        return high * (high + 1U) / 2U + low;
    }
    struct Row {
        float *values;
        unsigned row;
        PM_AVBD_LAYOUT_INLINE float &operator[](unsigned column) const noexcept {
            return values[index(row, column)];
        }
    };
    struct ConstRow {
        const float *values;
        unsigned row;
        PM_AVBD_LAYOUT_INLINE const float &operator[](
            unsigned column) const noexcept {
            return values[index(row, column)];
        }
    };
    PM_AVBD_LAYOUT_INLINE Row operator[](unsigned row) noexcept {
        return {values, row};
    }
    PM_AVBD_LAYOUT_INLINE ConstRow operator[](unsigned row) const noexcept {
        return {values, row};
    }
};

struct Block {
    Symmetric6 h{};
    float g[6]{};
};

// Linearized bilateral/limit/spring/motor row storage. The equations that
// consume this data are defined by the shared Slang implementation.
struct Row {
    Vector6 a{}, b{};
    Dual dual{};
    float error{}, velocity{};
    float lower{-maximum_penalty}, upper{maximum_penalty};
    float stiffness{maximum_penalty}, damping{};
    float beta{linear_beta};
    float reference_force{};
};

} // namespace parallel_mater::avbd

#undef PM_AVBD_LAYOUT_INLINE
