// SPDX-License-Identifier: MIT
#pragma once

// Scalar normal/tangent block shared by CUDA, Metal, and CPU fixtures.
#if defined(__METAL_VERSION__)
#define PM_CONTACT_FRICTION_INLINE inline
#define PM_CONTACT_FRICTION_NOEXCEPT
#elif defined(__CUDACC__)
#define PM_CONTACT_FRICTION_INLINE __host__ __device__ __forceinline__
#define PM_CONTACT_FRICTION_NOEXCEPT noexcept
#else
#define PM_CONTACT_FRICTION_INLINE inline
#define PM_CONTACT_FRICTION_NOEXCEPT noexcept
#endif

namespace parallel_mater::solver {
struct CoupledContactFriction {
    float normal{};
    float tangent{}; // Signed impulse along the positive tangent residual.
};

// Eliminate the normal equation before a tangent descent step. The adapter
// chooses the tangent from the residual AFTER applying the normal impulse.
// For SPD response M this preserves the normal gap and decreases the reduced
// tangent quadratic. Coulomb bounds use the compensated normal impulse, not
// the obsolete normal-only solution. This is one coupled coordinate step;
// it does not assert a fully converged two-dimensional sliding direction.
PM_CONTACT_FRICTION_INLINE CoupledContactFriction coupled_contact_friction(
    float normal_impulse, float normal_mass, float cross_mass,
    float tangent_mass, float tangent_residual, float friction)
    PM_CONTACT_FRICTION_NOEXCEPT {
    CoupledContactFriction result{normal_impulse, 0.0F};
    if (!(normal_impulse > 0.0F) || !(normal_mass > 1.0e-12F) ||
        !(tangent_residual > 0.0F) || !(friction > 0.0F)) return result;
    const float coupling = cross_mass / normal_mass;
    const float schur = tangent_mass - cross_mass * coupling;
    if (!(schur > 1.0e-12F)) return result;
    float magnitude = tangent_residual / schur;
    const float cone_denominator = 1.0F - friction * coupling;
    if (cone_denominator > 0.0F) {
        const float bound = friction * normal_impulse / cone_denominator;
        magnitude = magnitude < bound ? magnitude : bound;
    }
    result.normal += coupling * magnitude;
    result.tangent = -magnitude;
    return result;
}
} // namespace parallel_mater::solver

#undef PM_CONTACT_FRICTION_INLINE
#undef PM_CONTACT_FRICTION_NOEXCEPT
