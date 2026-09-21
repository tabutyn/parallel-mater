// SPDX-License-Identifier: MIT
#include "renderer_shared.hpp"

#include <optix_device.h>

using parallel_mater::gallery::optix_shared::HitData;
using parallel_mater::gallery::optix_shared::LaunchParameters;

extern "C" {
__constant__ LaunchParameters params;
}

namespace {

static __forceinline__ __device__ float3 add(float3 first, float3 second) {
    return make_float3(first.x + second.x, first.y + second.y, first.z + second.z);
}

static __forceinline__ __device__ float3 multiply(float3 value, float scale) {
    return make_float3(value.x * scale, value.y * scale, value.z * scale);
}

static __forceinline__ __device__ float dot(float3 first, float3 second) {
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

static __forceinline__ __device__ float3 normalize(float3 value) {
    return multiply(value, rsqrtf(fmaxf(dot(value, value), 1.0e-20F)));
}

static __forceinline__ __device__ void set_payload(float3 color) {
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}

static __forceinline__ __device__ unsigned char to_byte(float value) {
    const float gamma = sqrtf(fminf(fmaxf(value, 0.0F), 1.0F));
    return static_cast<unsigned char>(gamma * 255.0F + 0.5F);
}

} // namespace

extern "C" __global__ void __raygen__primary() {
    const uint3 launch = optixGetLaunchIndex();
    const float2 pixel =
        make_float2((static_cast<float>(launch.x) + 0.5F) /
                            static_cast<float>(params.width) *
                            2.0F -
                        1.0F,
                    (static_cast<float>(launch.y) + 0.5F) /
                            static_cast<float>(params.height) *
                            2.0F -
                        1.0F);
    const float3 direction = normalize(add(params.camera_w,
                                           add(multiply(params.camera_u, pixel.x),
                                               multiply(params.camera_v, pixel.y))));
    unsigned int red = __float_as_uint(0.0F);
    unsigned int green = __float_as_uint(0.0F);
    unsigned int blue = __float_as_uint(0.0F);
    optixTrace(params.scene, params.eye, direction, 0.001F, 1.0e16F, 0.0F,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0,
               red, green, blue);
    const float3 color = make_float3(__uint_as_float(red), __uint_as_float(green),
                                     __uint_as_float(blue));
    params.image[launch.y * params.width + launch.x] =
        make_uchar4(to_byte(color.x), to_byte(color.y), to_byte(color.z), 255U);
}

extern "C" __global__ void __miss__sky() {
    const float3 direction = normalize(optixGetWorldRayDirection());
    const float amount = 0.5F * (direction.y + 1.0F);
    set_payload(add(multiply(make_float3(0.06F, 0.075F, 0.105F),
                             1.0F - amount),
                    multiply(make_float3(0.34F, 0.5F, 0.75F), amount)));
}

extern "C" __global__ void __closesthit__surface() {
    const HitData &hit_data =
        *reinterpret_cast<const HitData *>(optixGetSbtDataPointer());
    const uint3 triangle = hit_data.triangles[optixGetPrimitiveIndex()];
    const float2 barycentric = optixGetTriangleBarycentrics();
    const float first_weight = 1.0F - barycentric.x - barycentric.y;
    float3 object_normal = add(
        multiply(hit_data.vertices[triangle.x].normal, first_weight),
        add(multiply(hit_data.vertices[triangle.y].normal, barycentric.x),
            multiply(hit_data.vertices[triangle.z].normal, barycentric.y)));
    float3 normal = normalize(optixTransformNormalFromObjectToWorldSpace(object_normal));
    const float3 ray_direction = normalize(optixGetWorldRayDirection());
    if (dot(normal, ray_direction) > 0.0F) {
        normal = multiply(normal, -1.0F);
    }
    const float3 hit_point =
        add(optixGetWorldRayOrigin(),
            multiply(optixGetWorldRayDirection(), optixGetRayTmax()));

    float3 base_color = hit_data.base_color;
    if (hit_data.checkerboard != 0U) {
        const int checker = (static_cast<int>(floorf(hit_point.x)) +
                             static_cast<int>(floorf(hit_point.z))) &
                            1;
        base_color = checker != 0 ? make_float3(0.08F, 0.1F, 0.13F)
                                  : make_float3(0.82F, 0.85F, 0.9F);
    }
    const float3 light = normalize(make_float3(-0.45F, 0.82F, 0.35F));
    const float diffuse = fmaxf(dot(normal, light), 0.0F);
    const float rim = powf(1.0F - fmaxf(-dot(normal, ray_direction), 0.0F), 3.0F);
    const float illumination = 0.16F + 0.78F * diffuse;
    set_payload(add(multiply(base_color, illumination),
                    multiply(make_float3(0.32F, 0.48F, 0.7F), 0.18F * rim)));
}
