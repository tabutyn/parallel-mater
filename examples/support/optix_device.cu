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

static __forceinline__ __device__ float3 subtract(float3 first, float3 second) {
    return make_float3(first.x - second.x, first.y - second.y, first.z - second.z);
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

static __forceinline__ __device__ float3 sky(float3 direction) {
    const float amount = 0.5F * (direction.y + 1.0F);
    return add(multiply(make_float3(0.06F, 0.075F, 0.105F), 1.0F - amount),
               multiply(make_float3(0.34F, 0.5F, 0.75F), amount));
}

// Trilinear interpolation of the particle-derived weighted-center field.
static __forceinline__ __device__ float surface_value(
    parallel_mater::gallery::optix_shared::FluidSurfaceView surface,
    float3 point) {
    const auto grid = surface.grid;
    if (!surface.values || grid.dimensions.x < 2U || grid.dimensions.y < 2U ||
        grid.dimensions.z < 2U) return 1.0e20F;
    const float3 q = make_float3(
        (point.x - grid.minimum.x) / grid.cell_size.x,
        (point.y - grid.minimum.y) / grid.cell_size.y,
        (point.z - grid.minimum.z) / grid.cell_size.z);
    if (q.x < 0.0F || q.y < 0.0F || q.z < 0.0F ||
        q.x > grid.dimensions.x - 1U || q.y > grid.dimensions.y - 1U ||
        q.z > grid.dimensions.z - 1U) return grid.support_radius;
    const unsigned x = min(grid.dimensions.x - 2U,
                           static_cast<unsigned>(floorf(q.x)));
    const unsigned y = min(grid.dimensions.y - 2U,
                           static_cast<unsigned>(floorf(q.y)));
    const unsigned z = min(grid.dimensions.z - 2U,
                           static_cast<unsigned>(floorf(q.z)));
    const float tx = fminf(1.0F, q.x - x);
    const float ty = fminf(1.0F, q.y - y);
    const float tz = fminf(1.0F, q.z - z);
    const unsigned row = grid.dimensions.x;
    const unsigned layer = row * grid.dimensions.y;
    const unsigned i = z * layer + y * row + x;
    const float *v = surface.values;
    const float a = v[i] + tx * (v[i + 1U] - v[i]);
    const float b = v[i + row] + tx * (v[i + row + 1U] - v[i + row]);
    const float c = v[i + layer] + tx * (v[i + layer + 1U] - v[i + layer]);
    const float d = v[i + layer + row] +
                    tx * (v[i + layer + row + 1U] - v[i + layer + row]);
    return (a + ty * (b - a)) * (1.0F - tz) +
           (c + ty * (d - c)) * tz;
}

static __forceinline__ __device__ bool trace_surface(
    parallel_mater::gallery::optix_shared::FluidSurfaceView surface,
    float3 eye, float3 direction, float opaque_depth,
    float &distance, float3 &normal) {
    const auto grid = surface.grid;
    if (!surface.values) return false;
    const float origin[3]{eye.x, eye.y, eye.z};
    const float ray[3]{direction.x, direction.y, direction.z};
    const float low[3]{grid.minimum.x, grid.minimum.y, grid.minimum.z};
    const float high[3]{
        low[0] + grid.cell_size.x * (grid.dimensions.x - 1U),
        low[1] + grid.cell_size.y * (grid.dimensions.y - 1U),
        low[2] + grid.cell_size.z * (grid.dimensions.z - 1U)};
    float enter = 0.001F;
    float leave = opaque_depth;
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(ray[axis]) < 1.0e-9F) {
            if (origin[axis] < low[axis] || origin[axis] > high[axis])
                return false;
            continue;
        }
        float first = (low[axis] - origin[axis]) / ray[axis];
        float last = (high[axis] - origin[axis]) / ray[axis];
        if (first > last) { const float tmp = first; first = last; last = tmp; }
        enter = fmaxf(enter, first);
        leave = fminf(leave, last);
        if (enter >= leave) return false;
    }
    const float step = 0.35F *
        fminf(grid.cell_size.x, fminf(grid.cell_size.y, grid.cell_size.z));
    float previous_t = enter;
    float previous = surface_value(surface,
        add(eye, multiply(direction, previous_t)));
    for (int iteration = 0; iteration < 1024 && previous_t < leave;
         ++iteration) {
        const float current_t = fminf(previous_t + step, leave);
        const float current = surface_value(surface,
            add(eye, multiply(direction, current_t)));
        if ((previous < 0.0F) != (current < 0.0F)) {
            float first = previous_t;
            float last = current_t;
            for (int refine = 0; refine < 12; ++refine) {
                const float mid = 0.5F * (first + last);
                const float value = surface_value(surface,
                    add(eye, multiply(direction, mid)));
                if ((value < 0.0F) == (previous < 0.0F)) first = mid;
                else last = mid;
            }
            distance = 0.5F * (first + last);
            const float3 p = add(eye, multiply(direction, distance));
            const float3 e = multiply(grid.cell_size, 0.8F);
            normal = normalize(make_float3(
                (surface_value(surface, add(p, make_float3(e.x, 0, 0))) -
                 surface_value(surface, add(p, make_float3(-e.x, 0, 0)))) / e.x,
                (surface_value(surface, add(p, make_float3(0, e.y, 0))) -
                 surface_value(surface, add(p, make_float3(0, -e.y, 0)))) / e.y,
                (surface_value(surface, add(p, make_float3(0, 0, e.z))) -
                 surface_value(surface, add(p, make_float3(0, 0, -e.z)))) / e.z));
            return true;
        }
        if (current_t >= leave) break;
        previous_t = current_t;
        previous = current;
    }
    return false;
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
    unsigned int depth = __float_as_uint(1.0e16F);
    optixTrace(params.scene, params.eye, direction, 0.001F, 1.0e16F, 0.0F,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0,
               red, green, blue, depth);
    float3 color = make_float3(__uint_as_float(red), __uint_as_float(green),
                               __uint_as_float(blue));
    float visible_depth = __uint_as_float(depth);
    float water_depth = 0.0F;
    float3 normal{};
    if (trace_surface(params.fluid, params.eye, direction, visible_depth,
                      water_depth, normal)) {
        if (dot(normal, direction) > 0.0F) normal = multiply(normal, -1.0F);
        const float facing = fmaxf(0.0F, -dot(normal, direction));
        const float fresnel = 0.02037F +
            0.97963F * powf(1.0F - facing, 5.0F);
        const float3 point = add(params.eye, multiply(direction, water_depth));
        const float3 reflected = normalize(subtract(
            direction, multiply(normal, 2.0F * dot(direction, normal))));
        const float eta = 1.0F / 1.333F;
        const float cosine = -dot(normal, direction);
        const float k = 1.0F - eta * eta * (1.0F - cosine * cosine);
        float3 transmission = color;
        float path = 0.6F;
        if (k > 0.0F) {
            const float3 refracted = normalize(add(multiply(direction, eta),
                multiply(normal, eta * cosine - sqrtf(k))));
            unsigned int tr = __float_as_uint(0.0F);
            unsigned int tg = __float_as_uint(0.0F);
            unsigned int tb = __float_as_uint(0.0F);
            unsigned int td = __float_as_uint(1.0e16F);
            optixTrace(params.scene, add(point, multiply(refracted, 0.002F)),
                       refracted, 0.001F, 1.0e16F, 0.0F,
                       OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                       0, 1, 0, tr, tg, tb, td);
            transmission = make_float3(__uint_as_float(tr), __uint_as_float(tg),
                                        __uint_as_float(tb));
            if (__uint_as_float(td) < 1.0e15F)
                path = fminf(2.0F, fmaxf(0.3F, __uint_as_float(td)));
        }
        const float3 absorption = make_float3(
            expf(-1.35F * path), expf(-0.34F * path), expf(-0.16F * path));
        const float haze = 0.35F + 0.45F * (1.0F - absorption.x);
        const float3 tint = make_float3(0.012F, 0.40F, 0.46F);
        transmission = add(multiply(make_float3(
            transmission.x * absorption.x,
            transmission.y * absorption.y,
            transmission.z * absorption.z), 1.0F - haze),
            multiply(tint, haze));
        const float glint = powf(fmaxf(0.0F, dot(reflected,
            normalize(make_float3(-0.48F, 0.84F, 0.34F)))), 80.0F);
        color = add(add(multiply(sky(reflected), fresnel),
                        multiply(transmission, 1.0F - fresnel)),
                    multiply(make_float3(0.62F, 0.91F, 1.0F),
                             0.38F * glint));
        visible_depth = water_depth;
    }
    params.image[launch.y * params.width + launch.x] =
        make_uchar4(to_byte(color.x), to_byte(color.y), to_byte(color.z), 255U);
    params.depth[launch.y * params.width + launch.x] = visible_depth;
}

extern "C" __global__ void __miss__sky() {
    const float3 direction = normalize(optixGetWorldRayDirection());
    set_payload(sky(direction));
}

extern "C" __global__ void __closesthit__surface() {
    optixSetPayload_3(__float_as_uint(optixGetRayTmax()));
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
