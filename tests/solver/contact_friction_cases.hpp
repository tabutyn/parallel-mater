// SPDX-License-Identifier: MIT
#pragma once

#include "reference/contact_friction.hpp"

#include <cmath>

#if defined(__CUDACC__)
#define PM_FRICTION_FIXTURE_INLINE __host__ __device__ inline
#else
#define PM_FRICTION_FIXTURE_INLINE inline
#endif

namespace contact_friction_cases {

constexpr unsigned count = 10U;

PM_FRICTION_FIXTURE_INLINE bool evaluate(unsigned index) noexcept {
    using parallel_mater::solver::coupled_contact_friction;
    float normal_mass = 9.26357F;
    float cross_mass = 18.3384F;
    float tangent_mass = 41.6964F;
    float normal = 0.0001F / normal_mass;
    float residual = 0.000207963F;
    float friction = 0.5F;

    switch (index) {
    case 1: friction = 0.39F; break;
    case 2: friction = 0.0F; break;
    case 3:
        normal_mass = 2.0F;
        cross_mass = 0.0F;
        tangent_mass = 3.0F;
        normal = 0.5F;
        residual = 0.3F;
        break;
    case 4:
        normal_mass = 2.0F;
        cross_mass = -1.0F;
        tangent_mass = 3.0F;
        normal = 0.2F;
        residual = 1.0F;
        friction = 0.4F;
        break;
    case 5:
        normal_mass = 1.0F;
        cross_mass = 0.999F;
        tangent_mass = 1.0F;
        normal = 0.001F;
        residual = 0.1F;
        break;
    case 6:
        normal_mass = 1.0F;
        cross_mass = 2.0F;
        tangent_mass = 5.0F;
        normal = 0.001F;
        residual = 0.1F;
        friction = 1.0F;
        break;
    case 7: residual = 0.0F; break;
    case 8:
        normal_mass = 1.0F;
        cross_mass = 1.0F;
        tangent_mass = 1.0F;
        break;
    case 9:
        // Sweep signs, mass ratios, and cone-active states with independent
        // normal-equation, reduced-energy, and updated-cone checks.
        for (unsigned fixture = 0U; fixture < 96U; ++fixture) {
            const float nn = 0.25F + float(fixture % 7U);
            const float nt = float(int(fixture % 9U) - 4) * 0.2F;
            const float tt = nt * nt / nn + 0.125F + float(fixture % 5U);
            const float lambda = 0.01F + float(fixture % 3U) * 0.1F;
            const float r = 0.02F + float(fixture % 11U) * 0.03F;
            const float mu = float(fixture % 6U) * 0.2F;
            const auto value = coupled_contact_friction(
                lambda, nn, nt, tt, r, mu);
            const float delta = value.normal - lambda;
            const float schur = tt - nt * nt / nn;
            if (fabsf(nn * delta + nt * value.tangent) > 2.0e-6F ||
                value.normal < 0.0F ||
                fabsf(value.tangent) > mu * value.normal + 2.0e-6F ||
                r * value.tangent +
                    0.5F * schur * value.tangent * value.tangent > 1.0e-7F)
                return false;
        }
        return true;
    default: break;
    }

    const auto value = coupled_contact_friction(
        normal, normal_mass, cross_mass, tangent_mass, residual, friction);
    const float normal_change = value.normal - normal;
    const float schur =
        tangent_mass - cross_mass * cross_mass / normal_mass;
    const float normal_error =
        normal_mass * normal_change + cross_mass * value.tangent;
    const float final_tangent = residual + cross_mass * normal_change +
        tangent_mass * value.tangent;
    if (fabsf(normal_error) > 1.0e-7F || value.normal < 0.0F ||
        fabsf(value.tangent) > friction * value.normal + 1.0e-7F ||
        residual * value.tangent +
            0.5F * schur * value.tangent * value.tangent > 1.0e-8F)
        return false;
    if (index == 0U || index == 3U || index == 6U)
        return fabsf(final_tangent) < 1.0e-6F;
    if (index == 1U || index == 4U || index == 5U)
        return final_tangent > 0.0F &&
            fabsf(-value.tangent - friction * value.normal) < 1.0e-7F;
    return value.normal == normal && value.tangent == 0.0F;
}

} // namespace contact_friction_cases

#undef PM_FRICTION_FIXTURE_INLINE
