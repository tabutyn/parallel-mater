// SPDX-License-Identifier: MIT
// Slang mirror of src/geometry_rope.cuh. Core geometry/solver kernels are
// defined below the shared rope ABI helpers as this migration advances.

import cuda_common;

public int rope_anchor_body(RopeData rope, uint node, int first, int last)
{
    return node == 0u ? first : (node + 1u == rope.count ? last : -1);
}

public bool rope_soft_anchor(RopeData rope, uint node)
{
    return (node == 0u && rope.soft_first) ||
           (node + 1u == rope.count && rope.soft_last);
}

public uint rope_soft_end(RopeData rope, uint node)
{
    return node == 0u ? 0u : 1u;
}

public void closest_segments(
    Vec3 p1, Vec3 q1, Vec3 p2, Vec3 q2, out Vec3 c1, out Vec3 c2)
{
    Vec3 d1 = subtract(q1, p1);
    Vec3 d2 = subtract(q2, p2);
    Vec3 r = subtract(p1, p2);
    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);
    float s = 0.0f;
    float t = 0.0f;
    float squaredEpsilon = kEpsilon * kEpsilon;
    if (a <= squaredEpsilon && e <= squaredEpsilon)
    {
        c1 = p1;
        c2 = p2;
        return;
    }
    if (a <= squaredEpsilon)
        t = clamp_scalar(f / e, 0.0f, 1.0f);
    else
    {
        float c = dot(d1, r);
        if (e <= squaredEpsilon)
            s = clamp_scalar(-c / a, 0.0f, 1.0f);
        else
        {
            float b = dot(d1, d2);
            float denominator = a * e - b * b;
            if (abs(denominator) > kEpsilon * a * e)
                s = clamp_scalar((b * f - c * e) / denominator, 0.0f, 1.0f);
            t = (b * s + f) / e;
            if (t < 0.0f)
            {
                t = 0.0f;
                s = clamp_scalar(-c / a, 0.0f, 1.0f);
            }
            else if (t > 1.0f)
            {
                t = 1.0f;
                s = clamp_scalar((b - c) / a, 0.0f, 1.0f);
            }
        }
    }
    c1 = add(p1, multiply(d1, s));
    c2 = add(p2, multiply(d2, t));
}
