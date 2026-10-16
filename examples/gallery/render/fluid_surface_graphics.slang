// SPDX-License-Identifier: MIT
// GPU presentation work only: public particle output -> field -> triangle list.
import fluid_surface_field;
#include "gallery_uniforms.slangh"
ByteAddressBuffer physics : register(t0);
RWByteAddressBuffer metadata : register(u0);
RWStructuredBuffer<uint> heads : register(u1);
RWStructuredBuffer<uint> nextParticle : register(u2);
RWStructuredBuffer<float> field : register(u3);
RWStructuredBuffer<float3> vertices : register(u4);
static const uint maxSamples = 128 * 128 * 128;

float3 particlePosition(uint index) { return asfloat(physics.Load3(draw.y + 16 + index * 48)); }
uint cellHash(int3 cell)
{
    uint3 bits = uint3(cell);
    return ((bits.x * 73856093u) ^ (bits.y * 19349663u) ^ (bits.z * 83492791u)) & (draw.z - 1);
}

[shader("compute")]
[numthreads(128,1,1)]
void surfaceClear(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x < draw.z) heads[thread.x] = 0xffffffffu;
    if (thread.x == 0) {
        metadata.Store4(0, uint4(0,1,0,0));
        metadata.Store(60, 0u); // overflow
    }
}

[shader("compute")]
[numthreads(128,1,1)]
void surfaceHash(uint3 thread : SV_DispatchThreadID)
{
    uint index = thread.x, count = physics.Load(draw.y);
    if (index >= count) return;
    uint bucket = cellHash(int3(floor(particlePosition(index) / settings.z)));
    uint previous;
    InterlockedExchange(heads[bucket], index, previous);
    nextParticle[index] = previous;
}

[shader("compute")]
[numthreads(1,1,1)]
void surfaceBounds()
{
    uint count = physics.Load(draw.y);
    float3 low = float3(3.402823466e38), high = -low;
    for (uint index = 0; index < count; ++index) {
        float3 point = particlePosition(index);
        low = min(low, point); high = max(high, point);
    }
    if (count == 0) { metadata.Store3(48, uint3(0)); return; }
    float3 extent = high - low + 2.0 * settings.z;
    float pitch = max(settings.x * 1.5, pow(extent.x * extent.y * extent.z / float(maxSamples), 1.0 / 3.0));
    uint3 dimensions;
    for (;;) {
        dimensions = clamp(uint3(ceil(extent / pitch)) + 1, uint3(2), uint3(256));
        if (dimensions.x * dimensions.y * dimensions.z <= maxSamples) break;
        pitch *= 1.025;
    }
    metadata.Store3(16, asuint(low - settings.z));
    metadata.Store3(32, asuint(extent / float3(dimensions - 1)));
    metadata.Store3(48, dimensions);
}

uint fieldIndex(uint3 point, uint3 dimensions)
{ return point.x + dimensions.x * (point.y + dimensions.y * point.z); }

[shader("compute")]
[numthreads(128,1,1)]
void surfaceField(uint3 thread : SV_DispatchThreadID)
{
    uint3 dimensions = metadata.Load3(48);
    uint index = thread.x;
    if (index >= dimensions.x * dimensions.y * dimensions.z) return;
    uint3 coordinate = uint3(index % dimensions.x, (index / dimensions.x) % dimensions.y,
        index / (dimensions.x * dimensions.y));
    float3 point = asfloat(metadata.Load3(16)) + float3(coordinate) * asfloat(metadata.Load3(32));
    int3 cell = int3(floor(point / settings.z));
    float4 sum = 0.0;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x) {
                int3 neighbor = cell + int3(x,y,z);
                for (uint item = heads[cellHash(neighbor)]; item != 0xffffffffu; item = nextParticle[item]) {
                    float3 position = particlePosition(item);
                    if (all(int3(floor(position / settings.z)) == neighbor))
                        sum += surfaceContribution(point - position, settings.z);
                }
            }
    field[index] = surfaceValue(sum, settings.z, settings.x);
}

void emitTriangle(float3 a, float3 b, float3 c)
{
    uint first;
    metadata.InterlockedAdd(0, 3u, first);
    if (first + 3 > draw.w) { metadata.Store(60, 1u); return; }
    float3 normal = cross(b-a, c-a);
    normal = dot(normal, normal) > 1.0e-18 ? normalize(normal) : float3(0,1,0);
    vertices[first*2] = a; vertices[first*2+1] = normal;
    vertices[first*2+2] = b; vertices[first*2+3] = normal;
    vertices[first*2+4] = c; vertices[first*2+5] = normal;
}

[shader("compute")]
[numthreads(128,1,1)]
void surfaceMesh(uint3 thread : SV_DispatchThreadID)
{
    uint3 dimensions = metadata.Load3(48);
    if (any(dimensions < 2)) return;
    uint3 cells = dimensions - 1;
    uint index = thread.x;
    if (index >= cells.x * cells.y * cells.z) return;
    uint3 base = uint3(index % cells.x, (index / cells.x) % cells.y, index / (cells.x*cells.y));
    float3 low = asfloat(metadata.Load3(16)), spacing = asfloat(metadata.Load3(32));
    float3 points[8]; float values[8];
    for (uint corner = 0; corner < 8; ++corner) {
        uint3 coordinate = base + uint3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        points[corner] = low + float3(coordinate) * spacing;
        values[corner] = field[fieldIndex(coordinate, dimensions)];
    }
    const uint4 tetrahedra[6] = { uint4(0,1,3,7), uint4(0,3,2,7), uint4(0,2,6,7),
        uint4(0,6,4,7), uint4(0,4,5,7), uint4(0,5,1,7) };
    for (uint tetra = 0; tetra < 6; ++tetra) {
        uint4 ids = tetrahedra[tetra];
        uint inside[4], outside[4], nInside = 0, nOutside = 0;
        for (uint i = 0; i < 4; ++i) {
            if (values[ids[i]] < 0) inside[nInside++] = ids[i];
            else outside[nOutside++] = ids[i];
        }
        if (nInside == 0 || nInside == 4) continue;
        float3 crossings[4];
        uint count = 0;
        for (uint a = 0; a < nInside; ++a)
            for (uint b = 0; b < nOutside; ++b) {
                uint first = inside[a], second = outside[b];
                float t = saturate(values[first] / (values[first] - values[second]));
                crossings[count++] = lerp(points[first], points[second], t);
            }
        emitTriangle(crossings[0], crossings[1], crossings[2]);
        if (count == 4) emitTriangle(crossings[1], crossings[3], crossings[2]);
    }
}

[shader("compute")]
[numthreads(1,1,1)]
void surfaceFinalize()
{ metadata.Store(0, min(metadata.Load(0), draw.w)); }
