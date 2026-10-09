// SPDX-License-Identifier: MIT
// Slang mirror of src/geometry_fluid.cuh. Geometry contact kernels follow the
// shared closest-point helpers used by the coupling modules.

import cuda_common;

public Vec3 fluid_closest_triangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c)
{
    Vec3 ab = subtract(b, a);
    Vec3 ac = subtract(c, a);
    Vec3 ap = subtract(p, a);
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    Vec3 bp = subtract(p, b);
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
        return add(a, multiply(ab, d1 / (d1 - d3)));
    Vec3 cp = subtract(p, c);
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
        return add(a, multiply(ac, d2 / (d2 - d6)));
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f)
        return add(b, multiply(subtract(c, b),
            (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    float denominator = va + vb + vc;
    return denominator > 1.0e-12f
        ? add(a, add(multiply(ab, vb / denominator),
                     multiply(ac, vc / denominator)))
        : a;
}

public Vec3 fluid_closest_triangle_barycentric(
    Vec3 p, Vec3 a, Vec3 b, Vec3 c, out Vec3 weights)
{
    Vec3 ab = subtract(b, a);
    Vec3 ac = subtract(c, a);
    Vec3 ap = subtract(p, a);
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f)
    {
        weights = {1.0f, 0.0f, 0.0f};
        return a;
    }
    Vec3 bp = subtract(p, b);
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3)
    {
        weights = {0.0f, 1.0f, 0.0f};
        return b;
    }
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        float v = d1 / (d1 - d3);
        weights = {1.0f - v, v, 0.0f};
        return add(a, multiply(ab, v));
    }
    Vec3 cp = subtract(p, c);
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6)
    {
        weights = {0.0f, 0.0f, 1.0f};
        return c;
    }
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        float w = d2 / (d2 - d6);
        weights = {1.0f - w, 0.0f, w};
        return add(a, multiply(ac, w));
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f)
    {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        weights = {0.0f, 1.0f - w, w};
        return add(b, multiply(subtract(c, b), w));
    }
    float denominator = va + vb + vc;
    if (denominator <= 1.0e-12f)
    {
        weights = {1.0f, 0.0f, 0.0f};
        return a;
    }
    float inverse = 1.0f / denominator;
    float v = vb * inverse;
    float w = vc * inverse;
    weights = {1.0f - v - w, v, w};
    return add(a, add(multiply(ab, v), multiply(ac, w)));
}

public void atomic_add(Vec3* destination, Vec3 value)
{
    pmAtomicAdd(&destination.x, value.x);
    pmAtomicAdd(&destination.y, value.y);
    pmAtomicAdd(&destination.z, value.z);
}
