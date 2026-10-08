// SPDX-License-Identifier: MIT
#include <metal_stdlib>

using namespace metal;

struct GalleryVertex {
    packed_float3 position;
    packed_float3 normal;
    packed_float3 color;
    packed_float2 uv;
    float checkerboard;
};

struct GalleryUniforms {
    float4x4 view_projection;
    float viewport_height;
    packed_float3 eye;
    packed_float3 camera_u;
    packed_float3 camera_v;
    packed_float3 camera_w;
};

struct RigidInstance {
    packed_float3 position;
    float4 orientation;
};

struct ParticleVertex {
    packed_float3 position;
    packed_float4 color;
    float radius;
};

struct GalleryVarying {
    float4 position [[position]];
    float3 world_position;
    float3 normal;
    float3 color;
    float2 uv;
    float checkerboard;
    float3 view_vector;
};

vertex GalleryVarying gallery_vertex(
    device const GalleryVertex *vertices [[buffer(0)]],
    constant GalleryUniforms &uniforms [[buffer(1)]],
    uint vertex_id [[vertex_id]]) {
    const device GalleryVertex &source = vertices[vertex_id];
    GalleryVarying output;
    output.world_position = float3(source.position);
    output.position = uniforms.view_projection *
                      float4(output.world_position, 1.0f);
    output.normal = normalize(float3(source.normal));
    output.color = float3(source.color);
    output.uv = float2(source.uv);
    output.checkerboard = source.checkerboard;
    output.view_vector = output.world_position - float3(uniforms.eye);
    return output;
}

static float3 rotate_quaternion(float4 quaternion, float3 value) {
    const float3 vector = quaternion.xyz;
    return value + 2.0f * cross(
        vector, cross(vector, value) + quaternion.w * value);
}

vertex GalleryVarying gallery_rigid_vertex(
    device const GalleryVertex *vertices [[buffer(0)]],
    constant GalleryUniforms &uniforms [[buffer(1)]],
    device const RigidInstance *instances [[buffer(2)]],
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]]) {
    const device GalleryVertex &source = vertices[vertex_id];
    const device RigidInstance &instance = instances[instance_id];
    const float4 orientation = instance.orientation;
    GalleryVarying output;
    output.world_position = float3(instance.position) +
        rotate_quaternion(orientation, float3(source.position));
    output.position = uniforms.view_projection *
        float4(output.world_position, 1.0f);
    output.normal = normalize(
        rotate_quaternion(orientation, float3(source.normal)));
    output.color = float3(source.color);
    output.uv = float2(source.uv);
    output.checkerboard = source.checkerboard;
    output.view_vector = output.world_position - float3(uniforms.eye);
    return output;
}

fragment float4 gallery_fragment(GalleryVarying input [[stage_in]]) {
    float3 color = input.color;
    if (input.checkerboard > 0.5f) {
        const int2 tile = int2(floor(input.world_position.xz));
        color = ((tile.x + tile.y) & 1) != 0
            ? float3(0.08f, 0.10f, 0.13f)
            : float3(0.82f, 0.85f, 0.90f);
    }
    const float3 ray_direction = normalize(input.view_vector);
    float3 normal = normalize(input.normal);
    if (dot(normal, ray_direction) > 0.0f) normal = -normal;
    const float3 light = normalize(float3(-0.45f, 0.82f, 0.35f));
    const float diffuse = max(dot(normal, light), 0.0f);
    const float rim = pow(1.0f - max(-dot(normal, ray_direction), 0.0f),
                          3.0f);
    color = color * (0.16f + 0.78f * diffuse) +
            float3(0.32f, 0.48f, 0.70f) * (0.18f * rim);
    return float4(sqrt(clamp(color, 0.0f, 1.0f)), 1.0f);
}

struct SkyVarying {
    float4 position [[position]];
    float3 direction;
};

vertex SkyVarying gallery_sky_vertex(
    constant GalleryUniforms &uniforms [[buffer(1)]],
    uint vertex_id [[vertex_id]]) {
    constexpr float2 positions[3] = {
        float2(-1.0f, -1.0f), float2(3.0f, -1.0f),
        float2(-1.0f, 3.0f)};
    const float2 position = positions[vertex_id];
    SkyVarying output;
    output.position = float4(position, 1.0f, 1.0f);
    output.direction = float3(uniforms.camera_w) +
        float3(uniforms.camera_u) * position.x +
        float3(uniforms.camera_v) * position.y;
    return output;
}

fragment float4 gallery_sky_fragment(SkyVarying input [[stage_in]]) {
    const float3 direction = normalize(input.direction);
    const float amount = 0.5f * (direction.y + 1.0f);
    const float3 color = mix(float3(0.06f, 0.075f, 0.105f),
                             float3(0.34f, 0.50f, 0.75f), amount);
    return float4(sqrt(clamp(color, 0.0f, 1.0f)), 1.0f);
}

struct ParticleVarying {
    float4 position [[position]];
    float4 color;
    float point_size [[point_size]];
    float center_distance;
    float radius;
};

vertex ParticleVarying gallery_particle_vertex(
    device const ParticleVertex *vertices [[buffer(0)]],
    constant GalleryUniforms &uniforms [[buffer(1)]],
    uint vertex_id [[vertex_id]]) {
    const device ParticleVertex &source = vertices[vertex_id];
    ParticleVarying output;
    output.position = uniforms.view_projection *
                      float4(float3(source.position), 1.0f);
    output.color = float4(source.color);
    output.center_distance = output.position.w;
    output.radius = source.radius;
    output.point_size = clamp(
        source.radius * uniforms.viewport_height *
            uniforms.view_projection[1][1] /
            max(output.center_distance, 0.05f),
        1.5f, 96.0f);
    return output;
}

struct ParticleOutput {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

fragment ParticleOutput gallery_particle_fragment(
    ParticleVarying input [[stage_in]],
    float2 point_coord [[point_coord]]) {
    const float2 centered = point_coord * 2.0f - 1.0f;
    const float radius_squared = dot(centered, centered);
    if (radius_squared > 1.0f) discard_fragment();
    const float z = sqrt(max(0.0f, 1.0f - radius_squared));
    const float diffuse = 0.32f + 0.68f *
        max(dot(normalize(float3(-centered.x, centered.y, z)),
                normalize(float3(-0.35f, 0.85f, 0.42f))), 0.0f);
    const float edge = smoothstep(1.0f, 0.72f, radius_squared);
    ParticleOutput output;
    output.color = input.color;
    output.color.rgb = pow(max(output.color.rgb * diffuse, 0.0f),
                           float3(1.0f / 2.2f));
    output.color.a *= edge;
    constexpr float near_plane = 0.05f;
    constexpr float far_plane = 120.0f;
    const float surface_distance = max(
        near_plane, input.center_distance - z * input.radius);
    output.depth = far_plane / (far_plane - near_plane) -
                   (far_plane * near_plane) /
                       ((far_plane - near_plane) * surface_distance);
    return output;
}

struct UiVertex {
    packed_float2 position;
    packed_float4 color;
};

struct UiUniforms {
    float2 viewport;
};

struct UiVarying {
    float4 position [[position]];
    float4 color;
};

vertex UiVarying gallery_ui_vertex(
    device const UiVertex *vertices [[buffer(0)]],
    constant UiUniforms &uniforms [[buffer(1)]],
    uint vertex_id [[vertex_id]]) {
    const device UiVertex &source = vertices[vertex_id];
    const float2 pixel = float2(source.position);
    UiVarying output;
    output.position = float4(
        pixel.x * 2.0f / uniforms.viewport.x - 1.0f,
        1.0f - pixel.y * 2.0f / uniforms.viewport.y,
        0.0f, 1.0f);
    output.color = float4(source.color);
    return output;
}

fragment float4 gallery_ui_fragment(UiVarying input [[stage_in]]) {
    return input.color;
}
