// SPDX-License-Identifier: MIT
// Common raster drawing of public pm* output records. No physics lives here.
#include "gallery_uniforms.slangh"
ByteAddressBuffer physics : register(t0);
StructuredBuffer<float3> surfaceVertices : register(t1);

struct VertexInput { float3 position : POSITION; float3 normal : NORMAL; };
struct FragmentInput
{
    float4 position : SV_Position;
    float3 world : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float foam : TEXCOORD3;
};

float3 rotate(float4 q, float3 v)
{
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

float4 project(float3 world)
{
    float3 relative = world - cameraEye.xyz;
    float z = dot(relative, cameraForward.xyz);
    // Infinite far plane, 1 cm near plane. D3D depth is [0, 1].
    return float4(dot(relative, cameraRight.xyz) / cameraRight.w,
                  dot(relative, cameraUp.xyz) / cameraUp.w, z - 0.01, z);
}

[shader("vertex")]
FragmentInput vertexMain(VertexInput input, uint vertex : SV_VertexID,
                         uint instance : SV_InstanceID)
{
    FragmentInput output = {};
    float3 world = input.position;
    float3 normal = input.normal;
    if (draw.x == 0)
    {
        uint offset = draw.y + draw.z * 52;
        float3 position = asfloat(physics.Load3(offset));
        float4 orientation = asfloat(physics.Load4(offset + 12));
        orientation = dot(orientation, orientation) > 0.1 ? normalize(orientation) : float4(0,0,0,1);
        uint previousOffset = draw.w + draw.z * 52;
        float4 previousOrientation = asfloat(physics.Load4(previousOffset + 12));
        if (dot(previousOrientation, previousOrientation) > 0.1)
        {
            if (dot(previousOrientation, orientation) < 0.0) previousOrientation = -previousOrientation;
            position = lerp(asfloat(physics.Load3(previousOffset)), position, saturate(settings.y));
            orientation = normalize(lerp(previousOrientation, orientation, saturate(settings.y)));
        }
        world = rotate(orientation, world) + position;
        normal = rotate(orientation, normal);
    }
    else if (draw.x == 1)
    {
        uint count = physics.Load(draw.y);
        if (vertex >= count) { output.position = float4(2,2,2,1); return output; }
        world = asfloat(physics.Load3(draw.y + 16 + vertex * 12));
    }
    else if (draw.x == 3)
    {
        world = surfaceVertices[vertex * 2];
        normal = surfaceVertices[vertex * 2 + 1];
    }
    else
    {
        uint count = physics.Load(draw.y);
        if (instance >= count) { output.position = float4(2,2,2,1); return output; }
        const float2 corners[6] = {float2(-1,-1),float2(1,-1),float2(-1,1),
                                  float2(-1,1),float2(1,-1),float2(1,1)};
        output.uv = corners[vertex % 6];
        uint offset = draw.y + 16 + instance * 48;
        float3 center = asfloat(physics.Load3(offset));
        world = center + settings.x * (cameraRight.xyz * output.uv.x + cameraUp.xyz * output.uv.y);
        normal = -cameraForward.xyz;
        output.foam = asfloat(physics.Load(offset + 40));
    }
    output.world = world;
    output.normal = normal;
    output.position = project(world);
    return output;
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input, bool front : SV_IsFrontFace) : SV_Target
{
    float3 normal;
    if (draw.x == 2)
    {
        float radius2 = dot(input.uv, input.uv);
        if (radius2 > 1.0) discard;
        normal = cameraRight.xyz * input.uv.x + cameraUp.xyz * input.uv.y -
                 cameraForward.xyz * sqrt(max(0.0, 1.0 - radius2));
    }
    else
    {
        normal = normalize(cross(ddx(input.world), ddy(input.world)));
        if (dot(normal, cameraEye.xyz - input.world) < 0.0) normal = -normal;
    }
    float shade = 0.28 + 0.72 * max(0.0, dot(normal, normalize(float3(0.349,0.814,0.465))));
    float3 color = draw.x == 2 ? lerp(baseColor.xyz, float3(0.92), saturate(input.foam)) : baseColor.xyz;
    return float4(color * shade, 1.0);
}
