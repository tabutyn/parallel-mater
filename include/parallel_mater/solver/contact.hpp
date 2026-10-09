// SPDX-License-Identifier: MIT
#pragma once

// Backend-neutral sequential-impulse contact row. No CUDA runtime / World API.
// Vector must be a float {x,y,z} aggregate. Adapter supplies point velocity,
// directional response (inverse_mass), and application of that response.
#include <cmath>

#if defined(__CUDACC__)
#define PM_CONTACT_INLINE __host__ __device__ __forceinline__
#else
#define PM_CONTACT_INLINE inline
#endif

namespace parallel_mater::solver {
constexpr float contact_epsilon = 1.0e-6F;
constexpr float surface_tolerance = 1.0e-5F;

// Convergence policy depends on contact workload, never on unrelated bodies,
// joints or a backend's launch geometry. Always warm-start before these passes.
PM_CONTACT_INLINE unsigned contact_iteration_budget(bool face_patch) noexcept {
    return face_patch ? 32U : 8U;
}

// Require two complete quiet sweeps, never a single quiet row. The caller
// reduces these changes over a connected contact/joint island and resets the
// state every substep. Bounds are deliberately below the contact skin.
struct ContactConvergence {
    unsigned quiet_sweeps{};
    bool finished{};

    PM_CONTACT_INLINE void observe(unsigned completed_passes, unsigned budget,
                                   float velocity_change, float position_change,
                                   float impulse_change = 0.0F) noexcept {
        const bool quiet = velocity_change >= 0.0F && velocity_change <= 1.0e-7F &&
                           position_change >= 0.0F && position_change <= 1.0e-8F &&
                           impulse_change >= 0.0F && impulse_change <= 1.0e-7F;
        quiet_sweeps = quiet ? quiet_sweeps + 1U : 0U;
        finished = completed_passes >= budget ||
                   (completed_passes >= 8U && quiet_sweeps >= 2U);
    }
};

template<class V> PM_CONTACT_INLINE V add(V a, V b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
template<class V> PM_CONTACT_INLINE V subtract(V a, V b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
template<class V> PM_CONTACT_INLINE V multiply(V a, float s) noexcept {
    return {a.x * s, a.y * s, a.z * s};
}
template<class V> PM_CONTACT_INLINE float dot(V a, V b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
template<class V> PM_CONTACT_INLINE V limit_length(V a, float limit) noexcept {
    const float squared = dot(a, a);
    if (squared <= limit * limit || squared <= contact_epsilon * contact_epsilon) return a;
#if defined(__CUDA_ARCH__)
    return multiply(a, limit * rsqrtf(squared));
#else
    return multiply(a, limit / sqrtf(squared));
#endif
}

struct ContactMaterial {
    float restitution{};
    float friction{};
    float friction_skin{};
    bool recover_penetration{};
};

PM_CONTACT_INLINE float contact_target_speed(
    float penetration, float incoming_speed, float restitution,
    float timestep, bool recover_penetration) noexcept {
    const float separation = fmaxf(0.0F, -penetration);
    const float dt = fmaxf(timestep, contact_epsilon);
    float target = separation > surface_tolerance ? -separation / dt : 0.0F;
    if (recover_penetration && penetration > 0.0F)
        target = fmaxf(target, 0.2F * penetration / dt);
    if (separation <= surface_tolerance && incoming_speed < 0.0F)
        target = fmaxf(target, -restitution * incoming_speed);
    return target;
}

template<class V> struct ContactImpulse {
    float normal{};
    V friction{};
};

// Minimal equation state; collision geometry and body-response bookkeeping
// belong in the adapter, not in each backend's copy of the row.
template<class V> struct ContactRow {
    V normal{};
    float penetration{};
    bool persistent{};
    float initial_normal_speed{};
    float accumulated_normal_impulse{};
    V accumulated_friction_impulse{};
};

// Persistent rows retain lambda across iterations/frames. Transient triangle
// rows have no stable feature identity: start at zero, never warm-start them.
// This is history policy, not a second set of solving equations.
template<class Row, class Adapter>
PM_CONTACT_INLINE auto solve_contact_velocity(
    Row &row, const ContactMaterial &material, float timestep,
    Adapter &adapter) noexcept -> ContactImpulse<decltype(row.normal)> {
    using V = decltype(row.normal);
    ContactImpulse<V> applied{};
    V relative_velocity = adapter.relative_velocity();
    const float speed = dot(relative_velocity, row.normal);
    const float target = contact_target_speed(row.penetration,
        row.persistent ? row.initial_normal_speed : speed,
        material.restitution, timestep, material.recover_penetration);
    if (!row.persistent && speed >= target) return applied;
    const auto normal_response = adapter.normal_response();
    if (normal_response.inverse_mass <= contact_epsilon) return applied;
    const float previous_normal = row.persistent ? row.accumulated_normal_impulse : 0.0F;
    const float normal = fmaxf(0.0F,
        previous_normal + (target - speed) / normal_response.inverse_mass);
    applied.normal = normal - previous_normal;
    adapter.apply(normal_response, applied.normal);

    const V previous_friction = row.persistent ? row.accumulated_friction_impulse : V{};
    const bool friction_active = fmaxf(0.0F, -row.penetration) <= material.friction_skin;
    V friction = friction_active ? previous_friction : V{};
    const float friction_limit = material.friction * normal;
    if (friction_active) {
        relative_velocity = adapter.relative_velocity();
        V tangent = subtract(relative_velocity, multiply(row.normal, dot(relative_velocity, row.normal)));
        const float tangent_squared = dot(tangent, tangent);
        if (tangent_squared > contact_epsilon * contact_epsilon) {
            // Directional response is quadratic in direction length. Using
            // the unnormalized tangent cancels the square root and both
            // normalization factors; it is the same sliding-friction row.
            // Transient rows retain their scalar clamp and floating-point
            // normalization order; they have no accumulated tangent lambda.
            if (!row.persistent) tangent = multiply(tangent, 1.0F / sqrtf(tangent_squared));
            const auto tangent_response = adapter.response(tangent);
            const float mass_epsilon = contact_epsilon * (row.persistent ? tangent_squared : 1.0F);
            if (tangent_response.inverse_mass > mass_epsilon) {
                const float speed = row.persistent ? tangent_squared : dot(relative_velocity, tangent);
                float impulse = -speed / tangent_response.inverse_mass;
                if (!row.persistent) impulse = fminf(friction_limit, fmaxf(-friction_limit, impulse));
                friction = add(friction, multiply(tangent, impulse));
            }
        }
    }
    if (row.persistent) friction = limit_length(friction, friction_limit);
    applied.friction = subtract(friction, previous_friction);
    // Application needs no effective-mass evaluation. Keep the tangent
    // response cache for unit directions, not arbitrary impulse vectors.
    adapter.apply_friction(applied.friction);
    if (row.persistent) {
        row.accumulated_normal_impulse = normal;
        row.accumulated_friction_impulse = friction;
    }
    return applied;
}
} // namespace parallel_mater::solver

#undef PM_CONTACT_INLINE
