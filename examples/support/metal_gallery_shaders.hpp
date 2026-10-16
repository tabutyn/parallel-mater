// SPDX-License-Identifier: MIT
// Graphics only. Physics remains in the shared Slang modules.
inline constexpr char gallery_metal_source[] = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Vertex { packed_float3 position; packed_float2 uv; uint output_index; };
struct Uniforms {
    float4 eye, right, up, forward, color;
    uint offset, stride, kind, count;
    uint paint_offset, paint_resolution, paint_count, padding;
    float radius, height, padding2, padding3;
};
struct Out {
    float4 position [[position]];
    float3 world;
    float2 uv;
    float point_size [[point_size]];
};
float3 read3(device const uchar *data) { return float3(*reinterpret_cast<device const packed_float3 *>(data)); }
float3 rotate(float4 q, float3 p) { return p + 2.f * cross(q.xyz, cross(q.xyz, p) + q.w * p); }
Out projected(float3 p, float2 uv, constant Uniforms &u) {
    float3 relative = p - u.eye.xyz;
    float z = dot(relative, u.forward.xyz);
    const float near = .01f, far = 10000.f;
    Out out;
    out.position = float4(dot(relative, u.right.xyz) / u.right.w,
        dot(relative, u.up.xyz) / u.up.w, z * far / (far-near) - near*far/(far-near), z);
    out.world = p; out.uv = uv;
    out.point_size = clamp(u.radius * u.height / max(.01f, z * u.up.w), 2.f, 24.f);
    return out;
}
vertex Out mesh_vertex(uint id [[vertex_id]], device const Vertex *vertices [[buffer(0)]],
    device const uchar *output [[buffer(1)]], constant Uniforms &u [[buffer(2)]]) {
    Vertex v = vertices[id];
    float3 p = float3(v.position);
    device const uchar *data = output + u.offset;
    if (u.kind == 0) {
        // Public PmRigidState: packed position, quaternion, velocities.
        device const float *q = reinterpret_cast<device const float *>(data + 12);
        p = rotate(normalize(float4(q[0],q[1],q[2],q[3])), p) + read3(data);
    } else {
        uint active = min(*reinterpret_cast<device const uint *>(data), u.count);
        if (v.output_index >= active) { Out out = {}; out.position = float4(2,2,2,1); return out; }
        p = read3(data + 16 + v.output_index * u.stride);
    }
    return projected(p, float2(v.uv), u);
}
vertex Out particle_vertex(uint id [[vertex_id]], device const uchar *output [[buffer(1)]],
    constant Uniforms &u [[buffer(2)]]) {
    device const uchar *data = output + u.offset;
    uint active = min(*reinterpret_cast<device const uint *>(data), u.count);
    if (id >= active) { Out out = {}; out.position = float4(2,2,2,1); out.point_size = 1; return out; }
    return projected(read3(data + 16 + id * u.stride), float2(0), u);
}
fragment float4 mesh_fragment(Out in [[stage_in]], device const uchar *output [[buffer(1)]],
    constant Uniforms &u [[buffer(2)]]) {
    float3 normal = normalize(cross(dfdx(in.world), dfdy(in.world)));
    float shade = .28f + .72f * abs(dot(normal, normalize(float3(.349f,.814f,.465f))));
    float3 color = u.color.rgb;
    if (u.paint_resolution) {
        uint2 xy = uint2(clamp(in.uv, 0.f, .999999f) * u.paint_resolution);
        uint index = xy.y * u.paint_resolution + xy.x;
        device const uint *paint = reinterpret_cast<device const uint *>(output + u.paint_offset);
        if (index < min(paint[0], u.paint_count)) {
            uint value = paint[4 + index];
            if (value) color = float3(value & 255, (value >> 8) & 255, (value >> 16) & 255) / 255.f;
        }
    }
    return float4(color * shade, 1);
}
fragment float4 particle_fragment(Out in [[stage_in]], float2 point [[point_coord]],
    constant Uniforms &u [[buffer(2)]]) {
    if (dot(point-.5f, point-.5f) > .25f) discard_fragment();
    return u.color;
}
struct OverlayOut { float4 position [[position]]; float2 uv; };
vertex OverlayOut overlay_vertex(uint id [[vertex_id]]) {
    const float2 p[3] = {float2(-1,-1),float2(3,-1),float2(-1,3)};
    return {float4(p[id],0,1), (p[id]+1)*.5f};
}
fragment float4 overlay_fragment(OverlayOut in [[stage_in]], texture2d<float> image [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    // Shared overlays/images are bottom-up; normalized Metal UVs start at top.
    return image.sample(s, in.uv);
}
)metal";
