// SPDX-License-Identifier: MIT
import fluid_surface_field;

[shader("compute")]
[numthreads(128, 1, 1)]
void surface_index_particles(uniform float3* positions, uniform uint count,
    uniform float inverseSupport, uniform uint64_t* keys, uniform uint* indices)
{
    uint index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (index >= count) return;
    keys[index] = surfaceCellKey(int3(floor(positions[index] * inverseSupport)));
    indices[index] = index;
}

uint surfaceLowerBound(uint64_t* keys, uint count, uint64_t key)
{
    uint first = 0;
    while (first < count) {
        uint middle = first + (count - first) / 2;
        if (keys[middle] < key) first = middle + 1;
        else count = middle;
    }
    return first;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void surface_build_field(uniform float3* positions, uniform uint count,
    uniform uint64_t* keys, uniform uint* indices, uniform SurfaceGrid grid,
    uniform uint samples, uniform float* values)
{
    uint index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (index >= samples) return;
    uint3 coordinate = uint3(index % grid.dimensions.x,
        (index / grid.dimensions.x) % grid.dimensions.y,
        index / (grid.dimensions.x * grid.dimensions.y));
    float3 point = grid.minimum + float3(coordinate) * grid.cell_size;
    int3 cell = int3(floor(point / grid.support_radius));
    float4 sum = 0.0;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x) {
                uint64_t key = surfaceCellKey(cell + int3(x,y,z));
                for (uint item = surfaceLowerBound(keys, count, key);
                     item < count && keys[item] == key; ++item)
                    sum += surfaceContribution(point - positions[indices[item]], grid.support_radius);
            }
    values[index] = surfaceValue(sum, grid.support_radius, grid.particle_radius);
}
