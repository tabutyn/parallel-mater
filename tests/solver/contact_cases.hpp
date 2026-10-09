// SPDX-License-Identifier: MIT
#pragma once
#include <parallel_mater/solver/contact.hpp>

#if defined(__CUDACC__)
#define PM_FIXTURE_INLINE __host__ __device__ inline
#else
#define PM_FIXTURE_INLINE inline
#endif

namespace contact_cases {
struct Vector { float x{}, y{}, z{}; };
using Row = parallel_mater::solver::ContactRow<Vector>;
struct Adapter {
    Vector velocity{};
    float inverse_mass{1.0F};
    Vector inverse_axes{1, 1, 1};
    struct Response { Vector direction; float inverse_mass; };
    PM_FIXTURE_INLINE Vector relative_velocity() const { return velocity; }
    PM_FIXTURE_INLINE Response response(Vector direction) const {
        return {direction, inverse_mass * (inverse_axes.x * direction.x * direction.x +
            inverse_axes.y * direction.y * direction.y + inverse_axes.z * direction.z * direction.z)};
    }
    PM_FIXTURE_INLINE Response normal_response() const { return response({0, 1, 0}); }
    PM_FIXTURE_INLINE void apply(Response response, float magnitude) {
        const auto impulse = parallel_mater::solver::multiply(response.direction, magnitude * inverse_mass);
        velocity = parallel_mater::solver::add(velocity,
            Vector{impulse.x * inverse_axes.x, impulse.y * inverse_axes.y, impulse.z * inverse_axes.z});
    }
    PM_FIXTURE_INLINE void apply_friction(Vector impulse) { apply({impulse, 0}, 1); }
};
struct Result {
    Vector velocity{}, friction{};
    float normal{}, accumulated{};
};
constexpr unsigned count = 25;

PM_FIXTURE_INLINE bool convergence_case(unsigned index) {
    parallel_mater::solver::ContactConvergence state{};
    switch (index) {
    case 14: // Never stop before the shared minimum.
        for (unsigned pass = 1; pass < 8; ++pass) state.observe(pass, 32, 0, 0);
        return !state.finished;
    case 15:
        state.observe(7, 32, 0, 0); state.observe(8, 32, 0, 0);
        return state.finished;
    case 16: // One quiet sweep is insufficient.
        state.observe(8, 32, 0, 0); return !state.finished;
    case 17: // Neighbor activity resets quiet history.
        state.observe(7, 32, 0, 0); state.observe(8, 32, 0.01F, 0);
        state.observe(9, 32, 0, 0); return !state.finished && state.quiet_sweeps == 1;
    case 18: // Position projection must also settle.
        state.observe(7, 32, 0, 0.001F); state.observe(8, 32, 0, 0.001F);
        return !state.finished;
    case 19: // Enforce a hard cap even for a nonconverging island.
        state.observe(32, 32, 1, 1); return state.finished;
    case 20: // Ordinary contact budget stays eight.
        state.observe(8, 8, 1, 1); return state.finished;
    case 21: // Invalid residuals must never count as quiet.
        state.observe(7, 32, -1, 0); state.observe(8, 32, -1, 0);
        return !state.finished && state.quiet_sweeps == 0;
    case 24: // Opposing impulses can hide behind unchanged body velocities.
        state.observe(7, 32, 0, 0, 0.01F); state.observe(8, 32, 0, 0, 0.01F);
        return !state.finished && state.quiet_sweeps == 0;
    }
    return false;
}

// The exact same inputs and expected outputs run as C++ on the host and as
// device code. New backends can execute these fixtures without the World API.
PM_FIXTURE_INLINE Result evaluate(unsigned index) {
    using namespace parallel_mater::solver;
    if ((index >= 14 && index < 22) || index == 24) return {{}, {}, convergence_case(index) ? 1.0F : 0.0F, 0};
    Row row{};
    row.normal = {0, 1, 0};
    Adapter adapter{{0, -2, 0}, 1};
    ContactMaterial material{0, 0, surface_tolerance, false};
    float timestep = 0.1F;
    switch (index) {
    case 0: break; // Inelastic stop.
    case 1: material.restitution = 0.5F; break;
    case 2: row.penetration = -0.1F; break; // Speculative gap permits -1 m/s.
    case 3: adapter.velocity.x = 4; material.friction = 0.5F; break;
    case 4: // Remove stale normal/friction support as persistent contact opens.
        row.persistent = true; row.penetration = -0.2F;
        row.accumulated_normal_impulse = 2; row.accumulated_friction_impulse = {1, 0, 0};
        adapter.velocity = {1, 1, 0}; break;
    case 5: adapter.inverse_mass = 0; break;
    case 6: material.recover_penetration = true; row.penetration = 0.1F;
        adapter.velocity = {}; break;
    case 7: row.persistent = true; row.initial_normal_speed = -2;
        material.restitution = 0.5F; break;
    case 8: adapter.velocity.y = 2; break; // Separating transient contact.
    case 9: row.penetration = -0.1F; timestep = 0; break; // Finite guarded dt.
    case 10: // Persistent static friction accumulates rather than restarting.
        row.persistent = true; row.accumulated_normal_impulse = 2;
        row.accumulated_friction_impulse = {-0.5F, 0, 0};
        adapter.velocity = {0.25F, 0, 0}; material.friction = 0.5F; break;
    case 11: // Project accumulated friction onto the updated Coulomb limit.
        row.persistent = true; row.accumulated_normal_impulse = 1;
        row.accumulated_friction_impulse = {-0.25F, 0, 0};
        adapter.velocity = {2, -1, 0}; material.friction = 0.5F; break;
    case 12: // Persistent static friction survives the numerical contact skin.
        row.persistent = true; row.penetration = -0.0005F;
        adapter.velocity = {2, -2, 0}; material.friction = 0.5F;
        material.friction_skin = 0.001F; break;
    case 13: adapter.inverse_mass = 2; break;
    case 22: case 23: // Anisotropic response must be quadratic in tangent length.
        adapter.velocity = {2, -2, 2}; adapter.inverse_axes = {1, 1, 4};
        material.friction = 10; row.persistent = index == 22; break;
    }
    const auto impulse = solve_contact_velocity(row, material, timestep, adapter);
    return {adapter.velocity, impulse.friction, impulse.normal, row.accumulated_normal_impulse};
}

PM_FIXTURE_INLINE Result expected(unsigned index) {
    if ((index >= 14 && index < 22) || index == 24) return {{}, {}, 1, 0};
    switch (index) {
    case 0: return {{0, 0, 0}, {}, 2, 0};
    case 1: return {{0, 1, 0}, {}, 3, 0};
    case 2: return {{0, -1, 0}, {}, 1, 0};
    case 3: return {{3, 0, 0}, {-1, 0, 0}, 2, 0};
    case 4: return {{0, -1, 0}, {-1, 0, 0}, -2, 0};
    case 5: return {{0, -2, 0}, {}, 0, 0};
    case 6: return {{0, 0.2F, 0}, {}, 0.2F, 0};
    case 7: return {{0, 1, 0}, {}, 3, 3};
    case 8: return {{0, 2, 0}, {}, 0, 0};
    case 10: return {{0, 0, 0}, {-0.25F, 0, 0}, 0, 2};
    case 11: return {{1.25F, 0, 0}, {-0.75F, 0, 0}, 1, 2};
    case 12: return {{1.0025F, -0.005F, 0}, {-0.9975F, 0, 0}, 1.995F, 1.995F};
    case 13: return {{0, 0, 0}, {}, 1, 0};
    case 22: return {{1.2F, 0, -1.2F}, {-0.8F, 0, -0.8F}, 2, 2};
    case 23: return {{1.2F, 0, -1.2F}, {-0.8F, 0, -0.8F}, 2, 0};
    default: return {{0, -2, 0}, {}, 0, 0};
    }
}

inline bool close(float a, float b) { return std::isfinite(a) && std::fabs(a - b) < 1.0e-5F; }
inline bool close(Vector a, Vector b) { return close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z); }
inline bool close(Result a, Result b) {
    return close(a.velocity, b.velocity) && close(a.friction, b.friction) &&
           close(a.normal, b.normal) && close(a.accumulated, b.accumulated);
}
} // namespace contact_cases
#undef PM_FIXTURE_INLINE
