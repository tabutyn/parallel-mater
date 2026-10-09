// SPDX-License-Identifier: MIT
#pragma once
#include <parallel_mater/solver/avbd.hpp>
#include <parallel_mater/solver/halfspace.hpp>
#include <cmath>

#if defined(__CUDACC__)
#define PM_AVBD_CASE_INLINE __host__ __device__ inline
#else
#define PM_AVBD_CASE_INLINE inline
#endif

namespace avbd_cases {
using namespace parallel_mater::avbd;

// Identical inputs and numerical operations compile for host and device.
// Validation uses an independent double-precision Gaussian elimination.
struct Result { float value[8]{}; bool solved{true}; };
constexpr unsigned block_count = 29;
constexpr unsigned scalar_count = 76;
constexpr unsigned count = block_count + scalar_count;

struct GeometryVector { float x{},y{},z{}; };
struct GeometryPlanes {
    GeometryVector normals[8]{};
    float offsets[8]{};
    GeometryVector corners[3]{};
    float margin{};
    unsigned count{};
    PM_AVBD_CASE_INLINE parallel_mater::solver::TriangleHalfspace<GeometryVector> operator[](unsigned face) const {
        parallel_mater::solver::TriangleHalfspace<GeometryVector> result{};
        result.normal = normals[face];
        for (unsigned corner = 0; corner < 3; ++corner) {
            const auto p = corners[corner], n = normals[face];
            result.depths[corner] = margin + offsets[face] - (n.x*p.x+n.y*p.y+n.z*p.z);
        }
        return result;
    }
};
PM_AVBD_CASE_INLINE GeometryPlanes geometry_box(GeometryVector center, GeometryVector half, float angle) {
    GeometryPlanes result{};
    result.count = 6;
    const float c = cosf(angle),s = sinf(angle);
    const GeometryVector normals[6]{{c,s,0},{-c,-s,0},{-s,c,0},{s,-c,0},{0,0,1},{0,0,-1}};
    for (unsigned face = 0; face < 6; ++face) {
        const auto n = normals[face];
        result.normals[face] = n;
        result.offsets[face] = n.x*center.x+n.y*center.y+n.z*center.z +
            (face < 2 ? half.x : face < 4 ? half.y : half.z);
    }
    return result;
}
PM_AVBD_CASE_INLINE Result evaluate_face_selection(unsigned scalar) {
    GeometryPlanes a{},b{};
    unsigned seed_a = 0, seed_b = 1;
    if (scalar == 70 || scalar == 71) {
        // The nearest opposing side faces request a ~1.1m escape. A common
        // top/bottom face pair permits the actual ~.0501m triangle escape.
        constexpr float angle = .002f;
        a = geometry_box({-.099f,0,0},{.1f,.05f,.1f},0);
        b = geometry_box({.099f*cosf(angle),-.099f*sinf(angle),0},{.1f,.05f,.1f},-angle);
        a.margin = b.margin = .0001f;
        if (scalar == 71) {
            a.corners[1] = {.0001f,.0002f,0};
            a.corners[2] = {-.0001f,.0001f,.0001f};
        }
    } else if (scalar == 72) {
        // Two overlapping octahedral sphere proxies. The initially selected
        // opposite faces are infeasible, but an alternative pair is nearby.
        constexpr float n = .5773502691896258f;
        a.count = b.count = 8;
        a.margin = b.margin = .0001f;
        for (unsigned face = 0; face < 8; ++face) {
            const GeometryVector normal{face&1 ? -n:n,face&2 ? -n:n,face&4 ? -n:n};
            a.normals[face] = b.normals[face] = normal;
            a.offsets[face] = .1f*n-.09f*normal.x;
            b.offsets[face] = .1f*n+.09f*normal.x;
        }
        seed_b = 7;
    } else {
        a = geometry_box({0,0,0},{1,1,1},0);
        b = geometry_box({10,0,0},{1,1,1},0);
        // All corners start outside A, but the triangle cuts through A.
        // Independent per-corner feature choices would incorrectly return0.
        a.corners[0] = {-2,0,0}; a.corners[1] = {2,0,0}; a.corners[2] = {0,2,0};
    }
    for (unsigned corner = 0; corner < 3; ++corner) b.corners[corner] = a.corners[corner];
    GeometryVector corrections[3]{},seed[3]{};
    float seed_cost = 0;
    bool seed_valid = true;
    const auto plane_a = a[seed_a],plane_b = b[seed_b];
    for (unsigned corner = 0; corner < 3; ++corner) {
        seed_valid = parallel_mater::solver::project_two_halfspaces(plane_a.normal,plane_a.depths[corner],
            plane_b.normal,plane_b.depths[corner],seed[corner]) && seed_valid;
        const auto d = seed[corner];
        seed_cost += d.x*d.x+d.y*d.y+d.z*d.z;
    }
    Result result{};
    result.solved = parallel_mater::solver::project_convex_pair_triangle(a,a.count,b,b.count,a[seed_a],b[seed_b],corrections);
    for (unsigned corner = 0; corner < 3; ++corner) {
        const auto d = corrections[corner];
        const float squared = d.x*d.x+d.y*d.y+d.z*d.z;
        result.value[0] += squared;
        result.value[1] = max_value(result.value[1],root(squared));
    }
    result.value[2] = 1;
    for (unsigned body = 0; body < 2; ++body) {
        const auto &planes = body == 0 ? a : b;
        float separation = -1e20f;
        for (unsigned face = 0; face < planes.count; ++face) {
            float minimum = 1e20f;
            const auto plane = planes[face];
            for (unsigned corner = 0; corner < 3; ++corner) {
                const auto d = corrections[corner],n = plane.normal;
                minimum = min_value(minimum,n.x*d.x+n.y*d.y+n.z*d.z-plane.depths[corner]);
            }
            separation = max_value(separation,minimum);
        }
        if (separation < -1e-6f) result.value[2] = 0;
    }
    result.value[3] = seed_valid ? seed_cost : -1;
    return result;
}

PM_AVBD_CASE_INLINE Block make_block(unsigned index) {
    Block block{};
    if (index >= 19) {
        if (index == 28) {
            // A cancellation-level negative eigenvalue, consistent with
            // roundoff in an intended PSD contact block, not real indefiniteness.
            for (unsigned i = 0; i < 6; ++i) block.h[i][i] = 1;
            block.h[1][0] = 1.00000011920928955078125f;
            block.g[0] = block.g[1] = 0.01f;
            return block;
        }
        const unsigned variant = (index-19)/3;
        const float penalties[] = {1e6f,1e8f,1e10f};
        const float penalty = penalties[(index-19)%3];
        const float translation = variant == 0 ? 100.0f : variant == 1 ? 57600.0f : 1.0f;
        const float rotation = variant == 0 ? 0.01f : variant == 1 ? 350.0f : 1e-4f;
        for (unsigned i = 0; i < 6; ++i) block.h[i][i] = i < 3 ? translation : rotation;
        if (variant == 0) {
            // This rounded rank-one block made the unscaled LDL^T reject a
            // physically positive-inertia body at the maximum penalty.
            add_row(block, {{0,1,0,0.013159f,0,-0.03762f}}, penalty, penalty*0.01f);
        } else {
            // One off-center contact's normal and tangents, r=(.08,-.09,.04).
            add_row(block, {{0,1,0,-0.04f,0,0.08f}}, penalty, penalty*0.012f);
            add_row(block, {{1,0,0,0,0.04f,0.09f}}, penalty, penalty*0.002f);
            add_row(block, {{0,0,1,-0.09f,-0.08f,0}}, penalty, penalty*-0.003f);
        }
        return block;
    }
    if (index == 18) {
        // Off-center contact couples translation Y to rotation Z.
        for (unsigned i = 0; i < 6; ++i) block.h[i][i] = i < 3 ? 2.0f : 0.5f;
        add_row(block, {{0, 1, 0, 0, 0, 2}}, 100, 3);
        return block;
    }
    const float scales[] = {1e-12f, 1e-6f, 1.0f, 1e6f, 1e12f, 1e18f};
    const float scale = scales[index % 6];
    const unsigned variant = index / 6;
    for (unsigned i = 0; i < 6; ++i) {
        const float inertia = variant == 0 ? float(i + 1) :
            (variant == 1 ? (i < 3 ? 1000.0f : 0.1f * float(i - 2)) : 0.3f * float(i + 1));
        block.h[i][i] = scale * inertia;
        block.g[i] = scale * (float(i) - 2.0f);
    }
    if (variant == 2) {
        block.h[4][3] = scale * 0.18f;
        block.h[5][3] = scale * -0.1f;
        block.h[5][4] = scale * 0.21f;
    }
    add_row(block, {{1, 2, -1, 0.5f, 0, 3}}, scale * 1.2f, scale * 0.7f);
    add_row(block, {{-0.5f, 0, 1, 2, -1, 0}}, scale * 3.0f, scale * -2.0f);
    add_row(block, {{0, 1, 0.25f, -1, 0.5f, 1}}, scale * 0.4f, scale * 0.9f);
    return block;
}

PM_AVBD_CASE_INLINE Result evaluate(unsigned index) {
    Result result{};
    if (index < block_count) {
        Vector6 delta{};
        result.solved = solve(make_block(index), delta);
        for (unsigned i = 0; i < 6; ++i) result.value[i] = delta.v[i];
        return result;
    }
    const unsigned scalar = index - block_count;
    if (scalar >= 70 && scalar < 74) return evaluate_face_selection(scalar);
    if (scalar >= 56) {
        struct Vector { float x,y,z; };
        Vector first{1,0,0},second{0,1,0},displacement{};
        float depth0=1,depth1=2;
        switch (scalar) {
        case 56: depth0=-1;depth1=-2;break;
        case 57: second={1,0,0};break;
        case 58: break;
        case 59: second={-1,0,0};depth1=-2;break;
        case 60: second={-1,0,0};depth1=1;break;
        case 61: second={.6f,.8f,0};depth1=0;break;
        case 62: second={-.6f,.8f,0};depth1=0;break;
        case 63: second={-.999f,.0447101778f,0};depth0=.01f;depth1=0;break;
        case 64: second={0,0,1};depth0=0;depth1=.005f;break;
        case 65: second={-.999f,.0447101778f,0};depth0=depth1=1e38f;break;
        case 66: depth0=INFINITY;break;
        case 67: second.x=NAN;break;
        case 68: depth0=1e38f;depth1=0;break;
        case 69: first={.8f,.6f,0};second={.8f,-.6f,0};depth0=depth1=3e38f;break;
        case 74: first=second={.707106769f,.707106769f,0};depth0=depth1=.05f;break;
        case 75: first={.707106769f,.707106769f,0};second={-.707106769f,-.707106769f,0};depth0=depth1=.05f;break;
        }
        result.solved=parallel_mater::solver::project_two_halfspaces(first,depth0,second,depth1,displacement);
        result.value[0]=displacement.x;result.value[1]=displacement.y;result.value[2]=displacement.z;
        return result;
    }
    Dual dual{};
    switch (scalar) {
    case 0: { Vector6 delta{}; result.solved = solve(Block{}, delta); break; }
    case 1: {
        Block block{}; Vector6 delta{};
        for (unsigned i = 0; i < 6; ++i) block.h[i][i] = 1;
        block.h[5][5] = -1;
        result.solved = solve(block, delta); break;
    }
    case 2: dual = update_dual({2, 10}, 0.1f, 100); break;
    case 3: dual = update_dual({0, 10}, -0.1f, 100); break;
    case 4: dual = update_dual({0, 10}, 0.1f, 100, -maximum_penalty, maximum_penalty, 15); break;
    case 5: dual = update_dual({0, 10}, 1, 100, -2, 2); break;
    case 6: dual = update_dual({0, 10}, -1, 100, -2, 2); break;
    case 7: dual = update_dual({0, 10}, -1, 100, 0, maximum_penalty); break;
    case 8: dual = update_dual({0, 10}, 0.1f, 100, 0, maximum_penalty); break;
    case 9: dual = update_dual({0, 10}, 0.2f, 100, -2, 2); break;
    case 10: dual = warm_start({10, 100}); break;
    case 11: dual = warm_start({0, 0.1f}); break;
    case 12: dual = warm_start({0, 100}, 0.25f); break;
    case 13: dual = update_dual({0, 0.25f}, 2, 100, -maximum_penalty, maximum_penalty, 0.25f); break;
    case 14: case 15: case 16: case 17: {
        const ContactForce input = scalar == 15 ? ContactForce{-1, 8, 6} :
            (scalar == 17 ? ContactForce{10, 1, -2} : ContactForce{10, 8, 6});
        const auto projected = project_contact(input, scalar == 16 ? 0.0f : 0.5f);
        result.value[0] = projected.normal;
        result.value[1] = projected.tangent0;
        result.value[2] = projected.tangent1;
        return result;
    }
    case 18: {
        Row row{};
        row.a = {{1, 0, 0, 0, 0, 2}}; row.b = {{-1, 0, 0, 0, 0, -3}};
        row.error = 0.25f;
        result.value[0] = row_error(row, {{1, 0, 0, 0, 0, 0.5f}}, {{0.5f, 0, 0, 0, 0, 0.25f}});
        return result;
    }
    case 19: { // Viscous damping contributes c/h to the Hessian, not c.
        Row row{}; Block block{};
        row.a = {{1, 0, 0, 0, 0, 0}}; row.b = {{-1, 0, 0, 0, 0, 0}};
        row.dual = {2, 10}; row.error = 0.25f; row.velocity = -1; row.damping = 3;
        accumulate(block, row, true, {{0.5f, 0, 0, 0, 0, 0}}, {{0.25f, 0, 0, 0, 0, 0}}, 0.5f);
        result.value[0] = block.h[0][0]; result.value[1] = block.g[0];
        return result;
    }
    case 20: { // Opposite endpoint contributes equal energy, opposite force.
        Row row{}; Block block{};
        row.a = {{1, 0, 0, 0, 0, 0}}; row.b = {{-1, 0, 0, 0, 0, 0}};
        row.dual = {2, 10}; row.error = 0.25f;
        accumulate(block, row, false, {}, {}, 0.5f);
        result.value[0] = block.h[0][0]; result.value[1] = block.g[0];
        return result;
    }
    case 21: { // Springs never accumulate a Lagrange multiplier.
        Row row{}; row.error = 0.2f; row.dual.penalty = 5; row.stiffness = 20; row.beta = 100;
        for (unsigned iteration = 0; iteration < 8; ++iteration) row = advance(row, {}, {});
        result.value[0] = row.dual.lambda; result.value[1] = row.dual.penalty;
        result.value[2] = row_force(row, {}, {}, 0.1f);
        return result;
    }
    case 22: { // Clamp force after spring and damping contributions.
        Row row{}; row.dual = {2, 10}; row.error = 1; row.damping = 100; row.velocity = 2;
        row.lower = -5; row.upper = 5;
        result.value[0] = row_force(row, {}, {}, 0.1f);
        row.velocity = -2; result.value[1] = row_force(row, {}, {}, 0.1f);
        return result;
    }
    case 23: { // Core stores Newtons; impulse conversion occurs at API edges.
        const auto support = update_dual({9, 10}, 0.1f, 100);
        result.value[0] = support.lambda;
        result.value[1] = support.lambda * 0.01f;
        result.value[2] = support.lambda * 0.02f;
        return result;
    }
    case 24: { // Zero authored stiffness must remain force-free.
        Row row{}; row.error = 10; row.stiffness = 0; row.dual = warm_start({}, 0);
        row = advance(row, {}, {});
        result.value[0] = row_force(row, {}, {}, 0.1f);
        result.value[1] = row.dual.penalty;
        return result;
    }
    case 25: dual = update_dual({0, maximum_penalty * 0.9f}, 1, maximum_penalty); break;
    case 26: {
        Dual support{5, 10};
        support = update_dual(support, -0.2f, 100, 0, maximum_penalty);
        result.value[0] = support.lambda;
        support = update_dual(support, -1, 100, 0, maximum_penalty);
        result.value[1] = support.lambda; result.value[2] = support.penalty;
        return result;
    }
    case 27: { // Corrective primal solves increments, dual keeps total force.
        Row row{}; Block block{};
        row.a = {{1,0,0,0,0,0}};
        row.dual = {10,100}; row.error = 0.02f; row.reference_force = 10;
        accumulate(block, row, true, {}, {}, 0.1f);
        row = advance(row, {}, {});
        result.value[0] = block.g[0]; result.value[1] = row.dual.lambda;
        result.value[2] = row.reference_force;
        return result;
    }
    case 28: { // Existing support must not be applied twice by correction.
        Row row{}; Block block{};
        row.a = {{1,0,0,0,0,0}};
        row.dual = {10,100}; row.reference_force = 10;
        accumulate(block, row, true, {}, {}, 0.1f);
        result.value[0] = block.g[0]; result.value[1] = block.h[0][0];
        result.value[2] = row_force(row, {}, {}, 0.1f);
        return result;
    }
    case 29: { // Clamp total force before subtracting its applied reference.
        Row row{}; Block block{};
        row.b = {{-1,0,0,0,0,0}};
        row.dual = {10,100}; row.error = 1; row.reference_force = 10; row.upper = 12;
        accumulate(block, row, false, {}, {}, 0.1f);
        row = advance(row, {}, {});
        result.value[0] = block.g[0]; result.value[1] = row.dual.lambda;
        result.value[2] = row.dual.penalty;
        return result;
    }
    case 30: case 31: case 32: case 33: case 34: case 35: case 36: case 37: {
        // Clamped curvature must not resist a direction with zero physical
        // force; active friction scales radially, not per tangent component.
        const ContactForce inputs[] = {{-1,8,6}, {0,0,0}, {10,8,6}, {10,0,0},
            {10,1,-2}, {10,8,-6}, {10,3,4}, {10,0,0}};
        const float friction = scalar == 32 || scalar == 33 ? 0.0f : 0.5f;
        const auto scale = contact_stiffness_scales(inputs[scalar-30], friction);
        result.value[0] = scale.normal;
        result.value[1] = scale.tangent0;
        result.value[2] = scale.tangent1;
        return result;
    }
    case 38: case 39: case 40: case 41: {
        Row row{}; Block block{};
        row.a = {{1,0,0,0,0,0}};
        const bool lower_limit = scalar == 38 || scalar == 40;
        row.error = lower_limit ? -1.0f : 1.0f;
        if (lower_limit) row.lower = 0;
        else row.upper = 0;
        row.dual.penalty = scalar < 40 ? 100.0f : 1e8f;
        if (scalar >= 40) {
            row.dual.lambda = lower_limit ? 100.0f : -100.0f;
            row.reference_force = row.dual.lambda;
        }
        accumulate(block, row, true, {}, {}, 0.01f);
        result.value[0] = block.h[0][0]; result.value[1] = block.g[0];
        result.value[2] = row_force(row, {}, {}, 0.01f);
        return result;
    }
    case 42: { // A dormant coupled limit must not block inertial free motion.
        Row row{}; Block block{}; Vector6 delta{};
        for (unsigned i = 0; i < 6; ++i) block.h[i][i] = 2;
        block.g[0] = -2; block.g[1] = -4; block.g[5] = -6;
        row.a = {{1,1,0,0,0,2}}; row.error = -100;
        row.dual = {100,1e8f}; row.lower = 0;
        accumulate(block, row, true, {}, {}, 0.01f);
        result.solved = solve(block, delta);
        result.value[0] = delta.v[0]; result.value[1] = delta.v[1]; result.value[2] = delta.v[5];
        return result;
    }
    case 43: case 44: case 49: {
        // Saturated two-sided motors keep a PSD stiffness majorant: a zero
        // derivative can Newton-jump through the opposite force cap and cycle.
        // Their gradient and published impulse remain physically bounded.
        Row row{}; Block block{};
        for (unsigned i = 0; i < 6; ++i) block.h[i][i] = i < 3 ? 2.0f : 0.5f;
        block.h[5][0] = 0.4f; block.g[1] = -6;
        row.a = {{1,0,0,0,0,2}}; row.dual.penalty = 1e8f;
        row.error = scalar == 44 ? -1000.0f : 1000.0f;
        row.lower = -10; row.upper = 10;
        row.reference_force = scalar == 49 ? 6.0f : 0.0f;
        accumulate(block, row, true, {}, {}, 0.01f);
        result.value[0] = block.h[0][0]; result.value[1] = block.h[5][5]; result.value[2] = block.h[5][0];
        result.value[3] = block.g[0]; result.value[4] = block.g[5]; result.value[5] = block.g[1];
        result.value[6] = row_force(row, {}, {}, 0.01f)*0.01f;
        return result;
    }
    case 45: { // Interior bounded rows retain implicit damping curvature.
        Row row{}; Block block{};
        row.a = {{1,0,0,0,0,0}}; row.b = {{-1,0,0,0,0,0}};
        row.dual = {2,10}; row.error = 0.25f; row.velocity = -1; row.damping = 3;
        row.lower = -10; row.upper = 10;
        const Vector6 a{{0.5f,0,0,0,0,0}}, b{{0.25f,0,0,0,0,0}};
        accumulate(block, row, true, a, b, 0.5f);
        result.value[0] = row_trial(row, a, b, 0.5f);
        result.value[1] = row_force(row, a, b, 0.5f);
        result.value[2] = block.h[0][0]; result.value[3] = block.g[0];
        return result;
    }
    case 46: { // Damping can deactivate an otherwise interior limit force.
        Row row{}; Block block{};
        row.a = {{1,0,0,0,0,0}};
        row.dual = {2,10}; row.error = 0.25f; row.velocity = -2; row.damping = 3;
        row.lower = 0;
        accumulate(block, row, true, {}, {}, 0.5f);
        result.value[0] = row_trial(row, {}, {}, 0.5f);
        result.value[1] = row_force(row, {}, {}, 0.5f);
        result.value[2] = block.h[0][0]; result.value[3] = block.g[0];
        return result;
    }
    case 47: case 48: { // Non-motor boundaries use strict interior curvature.
        Row row{}; Block block{};
        const float sign = scalar == 47 ? 1.0f : -1.0f;
        row.a = {{1,0,0,0,0,0}}; row.dual.penalty = 10; row.error = sign*0.5f;
        row.lower = scalar == 47 ? 0.0f : -5.0f;
        row.upper = scalar == 47 ? 5.0f : 0.0f;
        row.reference_force = sign;
        accumulate(block, row, true, {}, {}, 0.01f);
        result.value[0] = row_trial(row, {}, {}, 0.01f);
        result.value[1] = block.h[0][0]; result.value[2] = block.g[0];
        return result;
    }
    case 50: case 51: {
        // Repeated primal minimization of one fixed-dual motor energy. Zero
        // saturated curvature alternates across both caps on this fixture;
        // the majorant gives monotone progress even with only 1/4/10 updates.
        Row row{}; Vector6 position{};
        const float sign = scalar == 50 ? 1.0f : -1.0f;
        row.a = {{1,0,0,0,0,2}}; row.dual.penalty = 100; row.error = sign*0.01f;
        row.lower = -0.1f; row.upper = 0.1f;
        bool monotone = true; float previous_residual = 6.3f, previous_coordinate = 0;
        for (unsigned iteration = 0; iteration < 16; ++iteration) {
            Block block{}; Vector6 delta{};
            for (unsigned i = 0; i < 6; ++i) {
                block.h[i][i] = i < 3 ? 2.0f : 0.5f;
                block.g[i] = block.h[i][i]*position.v[i];
            }
            block.g[1] -= 6;
            accumulate(block, row, true, position, {}, 1);
            if (!solve(block, delta)) { result.solved = false; return result; }
            for (unsigned i = 0; i < 6; ++i) position.v[i] += delta.v[i];
            const float f = row_force(row, position, {}, 1);
            const float residual = abs_value(2*position.v[0]+f) + abs_value(2*position.v[1]-6)
                + abs_value(0.5f*position.v[5]+2*f);
            const float coordinate = sign*dot6(row.a, position);
            monotone = monotone && residual <= previous_residual+1e-6f && abs_value(f) <= 0.1f
                && coordinate <= previous_coordinate+1e-7f && coordinate >= -8.5f/851.0f-1e-7f;
            previous_residual = residual; previous_coordinate = coordinate;
        }
        result.value[0] = monotone ? 1.0f : 0.0f;
        result.value[1] = position.v[1];
        result.value[2] = row_force(row, position, {}, 1);
        result.value[3] = dot6(row.a, position);
        result.value[4] = previous_residual < 1e-5f ? 1.0f : 0.0f;
        return result;
    }
    case 52: { // The saturated motor majorant includes implicit damping/h.
        Row row{}; Block block{};
        row.a = {{1,0,0,0,0,0}}; row.dual.penalty = 10;
        row.error = 100; row.damping = 3; row.lower = -5; row.upper = 5;
        accumulate(block, row, true, {}, {}, 0.5f);
        result.value[0] = block.h[0][0]; result.value[1] = block.g[0];
        return result;
    }
    case 53: case 54: case 55: {
        Block block{}; Vector6 delta{};
        for (unsigned i = 0; i < 6; ++i) block.h[i][i] = 1;
        if (scalar == 53) block.g[0] = INFINITY;
        else if (scalar == 54) block.h[1][0] = INFINITY;
        else block.h[1][0] = 2; // Positive diagonals do not imply SPD.
        result.solved = solve(block, delta);
        return result;
    }
    }
    result.value[0] = dual.lambda; result.value[1] = dual.penalty;
    return result;
}

inline bool close(double actual, double expected, double relative = 2e-5, double absolute = 2e-6) {
    return std::isfinite(actual) && std::fabs(actual - expected) <= absolute + relative * std::fabs(expected);
}

inline bool reference_solve(Block block, double (&solution)[6]) {
    double a[6][7]{};
    for (unsigned i = 0; i < 6; ++i) {
        for (unsigned j = 0; j < 6; ++j) a[i][j] = block.h[i > j ? i : j][i > j ? j : i];
        a[i][6] = -double(block.g[i]);
    }
    // Independent Gaussian elimination with partial pivoting, not LDL^T.
    for (unsigned k = 0; k < 6; ++k) {
        unsigned pivot = k;
        for (unsigned i = k + 1; i < 6; ++i)
            if (std::fabs(a[i][k]) > std::fabs(a[pivot][k])) pivot = i;
        if (!(std::fabs(a[pivot][k]) > 0)) return false;
        for (unsigned j = k; j < 7; ++j) {
            const double temporary = a[k][j]; a[k][j] = a[pivot][j]; a[pivot][j] = temporary;
        }
        for (unsigned i = k + 1; i < 6; ++i) {
            const double multiplier = a[i][k] / a[k][k];
            for (unsigned j = k; j < 7; ++j) a[i][j] -= multiplier * a[k][j];
        }
    }
    for (int i = 5; i >= 0; --i) {
        double rhs = a[i][6];
        for (unsigned j = unsigned(i) + 1; j < 6; ++j) rhs -= a[i][j] * solution[j];
        solution[i] = rhs / a[i][i];
    }
    return true;
}

inline const char *name(unsigned index) {
    if (index < block_count) return index >= 19 ? "high-condition contact block" :
        index == 18 ? "off-center contact block" : "scaled coupled inertia block";
    const char *names[] = {"singular block", "indefinite block", "positive hard dual", "negative hard dual",
        "finite spring cap", "motor upper bound", "motor lower bound", "separated unilateral contact",
        "active unilateral contact", "exact force bound", "warm-start decay", "minimum penalty",
        "sub-unit spring stiffness", "sub-unit spring cap", "radial friction cone", "no contact adhesion",
        "frictionless contact", "static friction interior", "two-body row error", "damping Hessian",
        "second endpoint response", "spring does not harden", "force bounds include damping",
        "force versus impulse units", "zero stiffness", "maximum penalty", "support unloading",
        "corrective phase total dual", "corrective phase support reference", "corrective phase force clamp",
        "inactive contact curvature", "zero contact curvature", "frictionless sliding curvature",
        "frictionless zero tangent curvature", "static friction curvature", "sliding radial curvature",
        "friction boundary curvature", "static zero tangent curvature",
        "dormant lower limit curvature", "dormant upper limit curvature",
        "dormant lower limit history", "dormant upper limit history", "dormant coupled limit response",
        "saturated positive motor majorant", "saturated negative motor majorant", "interior damped row curvature",
        "damping-inactive limit curvature", "upper row boundary curvature", "lower row boundary curvature",
        "saturated motor reference force", "positive motor monotone primal", "negative motor monotone primal",
        "saturated damped motor majorant", "infinite block gradient", "infinite off-diagonal",
        "materially indefinite positive-diagonal block", "inactive halfspaces",
        "parallel halfspace dominance", "orthogonal halfspace projection",
        "opposing feasible halfspaces", "opposing infeasible halfspaces",
        "inactive second halfspace", "coupled active halfspaces",
        "near-opposing halfspace projection", "second active halfspace",
        "overflowing halfspace multipliers", "nonfinite halfspace depth",
        "nonfinite halfspace normal", "large finite halfspace projection",
        "overflowing halfspace displacement", "opposing thin-box point escape",
        "opposing thin-box triangle escape", "polyhedral sphere pinch escape",
        "common triangle separating faces", "rotated parallel halfspaces",
        "rotated opposing infeasible halfspaces"};
    return names[index - block_count];
}

inline bool validate(unsigned index, const Result &result) {
    if (index < block_count) {
        const Block block = make_block(index);
        if (index >= 19) {
            // Inertia can be rounded out of a float contact matrix. Do not
            // demand a nonexistent exact solution of an intended SPD system:
            // require a bounded descent step and small scaled backward error.
            if (!result.solved) return false;
            double gradient_step = 0, quadratic = 0, maximum_residual = 0, maximum_magnitude = 0;
            for (unsigned i = 0; i < 6; ++i) {
                if (!std::isfinite(result.value[i]) || std::fabs(result.value[i]) >= 0.5) return false;
                double product = 0, magnitude = std::fabs(double(block.g[i]));
                for (unsigned j = 0; j < 6; ++j) {
                    const double term = double(block.h[i > j ? i : j][i > j ? j : i])*result.value[j];
                    product += term; magnitude += std::fabs(term);
                }
                const double scale = std::sqrt(double(block.h[i][i]));
                maximum_residual = std::fmax(maximum_residual,std::fabs(product+block.g[i])/scale);
                maximum_magnitude = std::fmax(maximum_magnitude,magnitude/scale);
                gradient_step += double(block.g[i])*result.value[i];
                quadratic += double(result.value[i])*product;
            }
            return gradient_step < 0 && gradient_step+0.5*quadratic <= 1e-5*std::fabs(gradient_step) &&
                   maximum_residual <= 5e-5*maximum_magnitude;
        }
        double reference[6]{};
        if (!result.solved || !reference_solve(block, reference)) return false;
        for (unsigned i = 0; i < 6; ++i) {
            if (!close(result.value[i], reference[i])) return false;
            double residual = block.g[i], magnitude = std::fabs(double(block.g[i]));
            for (unsigned j = 0; j < 6; ++j) {
                const double term = double(block.h[i > j ? i : j][i > j ? j : i]) * result.value[j];
                residual += term; magnitude += std::fabs(term);
            }
            if (std::fabs(residual) > 2e-6 * magnitude + 1e-30) return false;
        }
        if (index == 18 && (!close(result.value[1], -1.5 / 851.0, 2e-5, 1e-8) ||
                            !close(result.value[5], -12.0 / 851.0, 2e-5, 1e-8))) return false;
        return true;
    }
    const unsigned scalar = index - block_count;
    if (scalar <= 1 || (scalar >= 53 && scalar <= 55) || scalar == 60) return !result.solved;
    if ((scalar >= 65 && scalar <= 67) || scalar == 69 || scalar == 75)
        return !result.solved && result.value[0] == 0 && result.value[1] == 0 && result.value[2] == 0;
    if (!result.solved) return false;
    if (scalar >= 70 && scalar < 74) {
        if (result.value[2] != 1) return false;
        if (scalar == 70)
            return close(result.value[0],3*.0501*.0501,1e-4,1e-7) &&
                   close(result.value[1],.0501,1e-4,1e-7) && result.value[3] > 3;
        if (scalar == 71) {
            // First corner activates both top faces; second needs only A's
            // top face, third only B's. Their common face pair remains fixed.
            const double c = std::cos(.002),s = std::sin(.002);
            const double first_x = .0501*(1-c)/s;
            const double third_depth = .0501-(-.0001*s+.0001*c);
            const double expected = .0501*.0501+first_x*first_x+.0499*.0499+third_depth*third_depth;
            return close(result.value[0],expected,1e-5,1e-8) &&
                   close(result.value[1],std::hypot(first_x,.0501),1e-5,1e-7) && result.value[3] > 3;
        }
        if (scalar == 72) {
            const double component = (.01+std::sqrt(3.0)*.0001)/2;
            return close(result.value[0],6*component*component,1e-4,1e-8) &&
                   close(result.value[1],std::sqrt(2.0)*component,1e-4,1e-7) && result.value[3] == -1;
        }
        return close(result.value[0],2) && close(result.value[1],1);
    }
    if (scalar >= 56) {
        if (scalar == 74)
            return close(result.value[0],.05/std::sqrt(2.0),2e-6,1e-8) &&
                   close(result.value[1],.05/std::sqrt(2.0),2e-6,1e-8) && result.value[2] == 0;
        if (scalar == 68)
            return close(result.value[0],1e38) && result.value[1] == 0 && result.value[2] == 0;
        const double expected[][3]={{0,0,0},{2,0,0},{1,2,0},{1,0,0},{0,0,0},
            {1,0,0},{1,.75,0},{.01,.00999/.0447101778,0},{0,0,.005}};
        for(unsigned component=0;component<3;++component)
            if(!close(result.value[component],expected[scalar-56][component],4e-5,1e-7))return false;
        return true;
    }
    // These expected values do not call implementation helpers.
    const double expected[][8] = {
        {0, 0, 0}, {0, 0, 0}, {3, 20, 0}, {-1, 20, 0}, {0, 15, 0}, {2, 10, 0}, {-2, 10, 0},
        {0, 10, 0}, {1, 20, 0}, {2, 10, 0}, {9.8901, 99.9, 0}, {0, 1, 0}, {0, 0.25, 0},
        {0, 0.25, 0}, {10, 4, 3}, {0, 0, 0}, {10, 0, 0}, {10, 1, -2}, {1, 0, 0},
        {16, 5.5, 0}, {10, -4.5, 0}, {0, 20, 4}, {5, -5, 0}, {10, 0.1, 0.2},
        {0, 0, 0}, {9e9, 1e10, 0}, {3, 0, 30}, {2,12,10}, {0,100,10}, {-2,12,100},
        {0,0,0}, {0,0,0}, {1,0,0}, {1,0,0}, {1,1,1}, {1,0.5,0.5}, {1,1,1}, {1,1,1},
        {0,0,0}, {0,0,0}, {0,-100,0}, {0,100,0}, {1,2,3},
        {1e8,4e8,2e8,10,20,-6,0.1}, {1e8,4e8,2e8,-10,-20,-6,-0.1},
        {5.5,5.5,16,5.5}, {-1.5,0,0,0}, {5,0,4}, {-5,0,-4}, {1e8,4e8,2e8,4,8,-6,0.1},
        {1,3,1.0/851.0,-8.5/851.0,1}, {1,3,-1.0/851.0,8.5/851.0,1}, {16,5}
    };
    for (unsigned i = 0; i < 8; ++i)
        if (!close(result.value[i], expected[scalar][i])) return false;
    return true;
}

inline bool equivalent(unsigned index, const Result &a, const Result &b) {
    if (a.solved != b.solved) return false;
    // Near-null directions in these deliberately ill-conditioned float
    // matrices amplify CPU/CUDA FMA rounding. Forward components need not
    // agree there: both platforms must instead satisfy the same finite,
    // bounded-descent and scaled-backward-error contract. Ordinary blocks
    // and every scalar fixture retain strict component-by-component parity.
    if (index >= 19 && index < block_count) return validate(index, a) && validate(index, b);
    for (unsigned i = 0; i < 8; ++i)
        if (!close(a.value[i], b.value[i])) return false;
    return true;
}
} // namespace avbd_cases
#undef PM_AVBD_CASE_INLINE
