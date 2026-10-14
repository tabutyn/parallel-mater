// SPDX-License-Identifier: MIT
#pragma once

#if defined(__CUDACC__)
#define PM_HALFSPACE_INLINE __host__ __device__ __forceinline__
#else
#define PM_HALFSPACE_INLINE inline
#endif

namespace parallel_mater::solver {

PM_HALFSPACE_INLINE bool halfspace_finite(float value) noexcept {
    constexpr float largest = 3.402823466e38F;
    return value >= -largest && value <= largest;
}
template<class V>
PM_HALFSPACE_INLINE bool halfspace_finite_vector(V value) noexcept {
    return halfspace_finite(value.x) && halfspace_finite(value.y) &&
           halfspace_finite(value.z);
}

template<class V>
PM_HALFSPACE_INLINE bool halfspace_contains(V normal, float depth, V displacement,
    float roundoff = 8.0F*1.1920928955078125e-7F) noexcept {
    const float terms[4]{normal.x*displacement.x,normal.y*displacement.y,
                         normal.z*displacement.z,depth};
    float scale = 0;
    for (float term : terms) {
        if (!halfspace_finite(term)) return false;
        const float absolute = term < 0 ? -term : term;
        if (absolute > scale) scale = absolute;
    }
    if (scale == 0) return true;
    float magnitude = 0;
    for (float term : terms) magnitude += (term < 0 ? -term : term)/scale;
    // Scaling the inequality also keeps its error bound finite for valid
    // large coordinates; an overflowing tolerance must never imply success.
    return terms[0]/scale+terms[1]/scale+terms[2]/scale >= depth/scale-roundoff*magnitude;
}

// Minimum-norm displacement satisfying n0.d >= depth0 and n1.d >= depth1.
// Normals must have unit length; depths may be negative (existing clearance).
// False means invalid input, nonfinite output, or opposing constraints that
// are infeasible/too nearly singular. Failure leaves a zero displacement.
// No iterative projection, extra contact pass, or velocity impulse is needed.
template<class V>
PM_HALFSPACE_INLINE bool project_two_halfspaces(
    V n0, float depth0, V n1, float depth1, V &displacement) noexcept {
    displacement = {};
    if (!halfspace_finite_vector(n0) || !halfspace_finite_vector(n1) ||
        !halfspace_finite(depth0) || !halfspace_finite(depth1)) return false;
    if (depth0 <= 0.0F && depth1 <= 0.0F) return true;
    const float norm0 = n0.x*n0.x+n0.y*n0.y+n0.z*n0.z;
    const float norm1 = n1.x*n1.x+n1.y*n1.y+n1.z*n1.z;
    if (!(norm0 > 0) || !(norm1 > 0) || !halfspace_finite(norm0) || !halfspace_finite(norm1)) return false;
    const float alignment = n0.x*n1.x + n0.y*n1.y + n0.z*n1.z;
    // Float-normalized rotated normals need not have exactly unit squared
    // length. Use the actual one-row Gram diagonal and verify both planes;
    // otherwise identical normals can wrongly reach the singular-pair case.
    if (depth0 >= 0.0F) {
        const float multiplier = depth0/norm0;
        const V candidate{n0.x*multiplier,n0.y*multiplier,n0.z*multiplier};
        if (halfspace_finite_vector(candidate) && halfspace_contains(n0,depth0,candidate) &&
            halfspace_contains(n1,depth1,candidate)) {
            displacement = candidate;
            return true;
        }
    }
    if (depth1 >= 0.0F) {
        const float multiplier = depth1/norm1;
        const V candidate{n1.x*multiplier,n1.y*multiplier,n1.z*multiplier};
        if (halfspace_finite_vector(candidate) && halfspace_contains(n0,depth0,candidate) &&
            halfspace_contains(n1,depth1,candidate)) {
            displacement = candidate;
            return true;
        }
    }
    const float difference = 1.0F-alignment;
    const float determinant = difference*(1.0F+alignment);
    if (!(determinant > 1.0e-6F)) return false;
    // Equal depths on nearly parallel faces must not lose their common
    // displacement to subtraction cancellation. These are the same 2x2
    // numerators, arranged around the small depth difference instead.
    const float lambda0 = (alignment > 0 ? (depth0-depth1)+difference*depth1
                                         : depth0-alignment*depth1)/determinant;
    const float lambda1 = (alignment > 0 ? (depth1-depth0)+difference*depth0
                                         : depth1-alignment*depth0)/determinant;
    if (!(lambda0 >= 0.0F && lambda1 >= 0.0F) ||
        !halfspace_finite(lambda0) || !halfspace_finite(lambda1)) return false;
    const V candidate{n0.x*lambda0+n1.x*lambda1,
                      n0.y*lambda0+n1.y*lambda1,
                      n0.z*lambda0+n1.z*lambda1};
    if (!halfspace_finite_vector(candidate)) return false;
    displacement = candidate;
    return true;
}

template<class V> struct TriangleHalfspace {
    V normal{};
    float depths[3]{};
};

template<class V>
PM_HALFSPACE_INLINE bool triangle_halfspace_lower_bound(
    const TriangleHalfspace<V> &plane, float &cost) noexcept {
    cost = 0;
    if (!halfspace_finite_vector(plane.normal)) return false;
    for (unsigned corner = 0; corner < 3; ++corner) {
        const float depth = plane.depths[corner];
        if (!halfspace_finite(depth)) return false;
        if (depth > 0) cost += depth*depth;
    }
    return halfspace_finite(cost);
}

template<class V>
PM_HALFSPACE_INLINE bool project_triangle_halfspace_pair(
    const TriangleHalfspace<V> &first, const TriangleHalfspace<V> &second,
    V (&corrections)[3], float &cost) noexcept {
    V candidate[3]{};
    float candidate_cost = 0;
    for (unsigned corner = 0; corner < 3; ++corner) {
        if (!project_two_halfspaces(first.normal, first.depths[corner],
                second.normal, second.depths[corner], candidate[corner])) return false;
        const auto d = candidate[corner];
        for (unsigned side = 0; side < 2; ++side) {
            const auto &plane = side == 0 ? first : second;
            constexpr float roundoff = 64.0F*1.1920928955078125e-7F;
            if (!halfspace_contains(plane.normal,plane.depths[corner],d,roundoff)) return false;
        }
        candidate_cost += d.x*d.x + d.y*d.y + d.z*d.z;
    }
    if (!halfspace_finite(candidate_cost)) return false;
    for (unsigned corner = 0; corner < 3; ++corner) corrections[corner] = candidate[corner];
    cost = candidate_cost;
    return true;
}

// A convex body's exterior is a union of outward face halfspaces. Select
// ONE face from each body for the whole triangle, minimizing the sum of its
// three squared corner displacements. Per-corner face selection is unsafe:
// vertices can escape on opposite sides while their connecting edge remains
// inside a solid. This searches conservative triangle-separating face pairs,
// not every possible separating configuration of two arbitrary polyhedra.
// Sources expose TriangleHalfspace<V> operator[](unsigned); adapters perform
// local/world transforms without allocating another copy of mesh planes.
template<class V, class FirstPlanes, class SecondPlanes>
PM_HALFSPACE_INLINE bool project_convex_pair_triangle(
    const FirstPlanes &first, unsigned first_count,
    const SecondPlanes &second, unsigned second_count,
    const TriangleHalfspace<V> &seed_first, const TriangleHalfspace<V> &seed_second,
    V (&corrections)[3]) noexcept {
    for (unsigned corner = 0; corner < 3; ++corner) corrections[corner] = {};
    float best_cost = 3.402823466e38F;
    bool found = project_triangle_halfspace_pair(seed_first,seed_second,corrections,best_cost);
    if (found && best_cost == 0) return true;
    for (unsigned a = 0; a < first_count; ++a) {
        const auto plane_a = first[a];
        float lower_a = 0;
        if (!triangle_halfspace_lower_bound(plane_a,lower_a) || (found && lower_a >= best_cost)) continue;
        for (unsigned b = 0; b < second_count; ++b) {
            const auto plane_b = second[b];
            float lower_b = 0;
            if (!triangle_halfspace_lower_bound(plane_b,lower_b) || (found && lower_b >= best_cost)) continue;
            V candidate[3]{};
            float candidate_cost = 0;
            if (!project_triangle_halfspace_pair(plane_a,plane_b,candidate,candidate_cost) ||
                (found && candidate_cost >= best_cost)) continue;
            for (unsigned corner = 0; corner < 3; ++corner) corrections[corner] = candidate[corner];
            best_cost = candidate_cost;
            found = true;
            if (best_cost == 0) return true;
        }
    }
    return found;
}

} // namespace parallel_mater::solver
#undef PM_HALFSPACE_INLINE
