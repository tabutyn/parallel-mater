// SPDX-License-Identifier: MIT
cbuffer DrawConstants : register(b0) {
    row_major float4x4 model_view_projection;
    row_major float4x4 model;
    float4 base_color;
    uint checkerboard;
    float3 padding;
};

struct VertexInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};
struct VertexOutput {
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

VertexOutput gallery_vs(VertexInput input) {
    VertexOutput output;
    output.position = mul(float4(input.position, 1.0f), model_view_projection);
    output.normal = normalize(mul(float4(input.normal, 0.0f), model).xyz);
    output.uv = input.uv;
    return output;
}

float4 gallery_ps(VertexOutput input) : SV_Target0 {
    const float3 light = normalize(float3(-0.45f, 0.82f, -0.35f));
    const float diffuse = 0.22f + 0.78f * saturate(dot(normalize(input.normal), light));
    float shade = 1.0f;
    if (checkerboard != 0) {
        const int2 tile = int2(floor(input.uv * 12.0f));
        shade = ((tile.x + tile.y) & 1) == 0 ? 1.0f : 0.42f;
    }
    return float4(base_color.rgb * diffuse * shade, base_color.a);
}

struct OverlayInput { float2 position : POSITION; float4 color : COLOR; };
struct OverlayOutput { float4 position : SV_Position; float4 color : COLOR; };
OverlayOutput gallery_ui_vs(OverlayInput input) {
    OverlayOutput output;
    output.position=float4(input.position,0,1);
    output.color=input.color;
    return output;
}
float4 gallery_ui_ps(OverlayOutput input) : SV_Target0 { return input.color; }
