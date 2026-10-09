// SPDX-License-Identifier: MIT
// Shared production standalone-fluid kernels. Couplings live elsewhere.

struct PMPackedVec3
{
    float x;
    float y;
    float z;
};

struct PMFluidConstants
{
    float timestep;
    PMPackedVec3 gravity;
    uint count;
    float particleRadius;
    float supportRadius;
    float repulsion;
    float viscosity;
    float normalDamping;
    float velocityDamping;
    float maximumSpeed;
    float maximumPairAcceleration;
    float restDensity;
    uint maximumNeighbors;
};

struct PMParticleMetadata
{
    uint count;
    uint nextStableId;
    uint revision;
    uint capacity;
    uint64_t emitted;
    uint64_t destroyed;
    uint64_t boiled;
};

#if PM_TARGET_METAL
[[vk::binding(0, 0)]] RWStructuredBuffer<PMPackedVec3> positions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<PMPackedVec3> velocities : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<PMPackedVec3> accelerations : register(u2);
[[vk::binding(4, 0)]] RWStructuredBuffer<float> foam : register(u4);
[[vk::binding(6, 0)]] ConstantBuffer<PMFluidConstants> constantsBuffer : register(b6);
[[vk::binding(7, 0)]] StructuredBuffer<PMParticleMetadata> metadata : register(t7);
[[vk::binding(8, 0)]] RWStructuredBuffer<PMPackedVec3> previous : register(u8);
[[vk::binding(9, 0)]] RWStructuredBuffer<float> foamSources : register(u9);
[[vk::binding(10, 0)]] RWStructuredBuffer<uint64_t> cellKeys : register(u10);
[[vk::binding(12, 0)]] RWStructuredBuffer<uint> sortedIndices : register(u12);
[[vk::binding(14, 0)]] RWStructuredBuffer<uint> neighborOverflow : register(u14);
[[vk::binding(15, 0)]] RWStructuredBuffer<uint> maximumNeighborCount : register(u15);
#endif

static const int kFluidCellBias = 1 << 20;
static const uint64_t kFluidEmptyCell = ~uint64_t(0);

float3 pmLoad(PMPackedVec3 value)
{
    return float3(value.x, value.y, value.z);
}

PMPackedVec3 pmStore(float3 value)
{
    PMPackedVec3 result = { value.x, value.y, value.z };
    return result;
}

float3 pmLimit(float3 value, float maximumLength)
{
    float squaredLength = dot(value, value);
    float squaredMaximum = maximumLength * maximumLength;
    return squaredLength > squaredMaximum
        ? value * (maximumLength * rsqrt(squaredLength))
        : value;
}

uint64_t pmFluidCellKey(int x, int y, int z)
{
    x = clamp(x, -kFluidCellBias, kFluidCellBias - 1);
    y = clamp(y, -kFluidCellBias, kFluidCellBias - 1);
    z = clamp(z, -kFluidCellBias, kFluidCellBias - 1);
    return (uint64_t(x + kFluidCellBias) << 42u) |
           (uint64_t(y + kFluidCellBias) << 21u) |
           uint64_t(z + kFluidCellBias);
}

uint pmFluidLowerBound(RWStructuredBuffer<uint64_t> keys,
                       uint size, uint64_t key)
{
    uint lower = 0u;
    uint upper = size;
    while (lower < upper)
    {
        uint middle = lower + (upper - lower) / 2u;
        if (keys[middle] < key)
            lower = middle + 1u;
        else
            upper = middle;
    }
    return lower;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_fluid_cell_keys(
    uint3 dispatchThreadId : SV_DispatchThreadID
#if PM_TARGET_CUDA
    , uniform StructuredBuffer<PMPackedVec3> positions
    , uniform StructuredBuffer<uint> countBuffer
    , uniform uint capacity
    , uniform float inverseRadius
    , uniform RWStructuredBuffer<uint64_t> cellKeys
    , uniform RWStructuredBuffer<uint> sortedIndices
#endif
)
{
    uint index = dispatchThreadId.x;
#if PM_TARGET_METAL
    uint count = metadata[0].count;
    uint capacity = metadata[0].capacity;
    float inverseRadius = 1.0f / constantsBuffer.supportRadius;
#else
    uint count = countBuffer[0];
#endif
    if (index >= capacity)
        return;
    if (index >= count)
    {
#if PM_TARGET_CUDA
        sortedIndices[index] = index;
        cellKeys[index] = kFluidEmptyCell;
#endif
        return;
    }
    sortedIndices[index] = index;
    float3 scaled = floor(pmLoad(positions[index]) * inverseRadius);
    float minimumCell = float(-kFluidCellBias);
    float maximumCell = float(kFluidCellBias - 1);
    int3 cell = int3(clamp(scaled, minimumCell.xxx, maximumCell.xxx));
    cellKeys[index] = pmFluidCellKey(cell.x, cell.y, cell.z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_fluid_forces(
    uint3 dispatchThreadId : SV_DispatchThreadID
#if PM_TARGET_CUDA
    , uniform StructuredBuffer<PMPackedVec3> positions
    , uniform StructuredBuffer<PMPackedVec3> velocities
    , uniform RWStructuredBuffer<PMPackedVec3> accelerations
    , uniform StructuredBuffer<float> foam
    , uniform StructuredBuffer<uint> countBuffer
    , uniform PMFluidConstants constants
    , uniform PMPackedVec3 upValue
    , uniform RWStructuredBuffer<float> foamSources
    , uniform RWStructuredBuffer<uint64_t> cellKeys
    , uniform StructuredBuffer<uint> sortedIndices
    , uniform RWStructuredBuffer<uint> neighborOverflow
    , uniform RWStructuredBuffer<uint> maximumNeighborCount
#endif
)
{
    uint index = dispatchThreadId.x;
#if PM_TARGET_METAL
    PMFluidConstants constants = constantsBuffer;
    uint count = metadata[0].count;
    float3 gravity = pmLoad(constants.gravity);
    float3 up = dot(gravity, gravity) > 1.0e-12f
        ? -normalize(gravity)
        : float3(0.0f, 1.0f, 0.0f);
#else
    uint count = countBuffer[0];
    float3 up = pmLoad(upValue);
#endif
    if (index >= count)
        return;

    float3 position = pmLoad(positions[index]);
    float3 velocity = pmLoad(velocities[index]);
    float3 acceleration = 0.0f;
    float3 outward = 0.0f;
    float weight = 0.0f;
    float relativeSpeedSquared = 0.0f;
    float neighboringFoam = 0.0f;
    uint neighbors = 0u;
    float supportSquared = constants.supportRadius * constants.supportRadius;
    float inverseRadius = 1.0f / constants.supportRadius;
    float3 scaled = floor(position * inverseRadius);
    int3 center = int3(clamp(scaled,
        float(-kFluidCellBias).xxx, float(kFluidCellBias - 1).xxx));

    for (int dz = -1; dz <= 1; ++dz)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                int3 cell = center + int3(dx, dy, dz);
                if (any(cell < int3(-kFluidCellBias)) ||
                    any(cell >= int3(kFluidCellBias)))
                    continue;
                uint64_t key = pmFluidCellKey(cell.x, cell.y, cell.z);
                for (uint item = pmFluidLowerBound(cellKeys, count, key);
                     item < count && cellKeys[item] == key; ++item)
                {
                    uint other = sortedIndices[item];
                    if (other == index)
                        continue;
                    float3 delta = position - pmLoad(positions[other]);
                    float distanceSquared = dot(delta, delta);
                    if (distanceSquared >= supportSquared)
                        continue;
                    ++neighbors;
                    float distance = sqrt(max(distanceSquared, 1.0e-12f));
                    float3 direction = distanceSquared > 1.0e-12f
                        ? delta / distance
                        : (index < other
                            ? float3(-1.0f, 0.0f, 0.0f)
                            : float3(1.0f, 0.0f, 0.0f));
                    float q = 1.0f - distance * inverseRadius;
                    outward += direction * q;
                    weight += q;
                    float3 relativeVelocity =
                        pmLoad(velocities[other]) - velocity;
                    relativeSpeedSquared += dot(relativeVelocity,
                                                relativeVelocity) * q;
                    neighboringFoam = max(neighboringFoam, foam[other] * q);
                    float radialSpeed = dot(relativeVelocity, direction);
                    acceleration += direction *
                        (constants.repulsion *
                             (1000.0f / constants.restDensity) * q * q +
                         constants.normalDamping * radialSpeed);
                    acceleration += relativeVelocity *
                        (constants.viscosity * q);
                }
            }
        }
    }

    InterlockedMax(maximumNeighborCount[0], neighbors);
    if (neighbors > constants.maximumNeighbors)
        InterlockedAdd(neighborOverflow[0], 1u);
    if (constants.maximumPairAcceleration > 0.0f)
        acceleration = pmLimit(acceleration,
                               constants.maximumPairAcceleration);
    accelerations[index] = pmStore(acceleration);
    float exposure = length(outward) / max(weight, 1.0e-6f);
    float3 outwardDirection = length(outward) > 1.0e-12f
        ? normalize(outward)
        : up;
    float upward = max(0.0f, dot(outwardDirection, up));
    float agitation = sqrt(relativeSpeedSquared / max(weight, 1.0e-6f));
    foamSources[index] = max(
        clamp((exposure - 0.12f) * 2.0f, 0.0f, 1.0f) * upward *
            clamp((agitation - 0.15f) * 1.5f, 0.0f, 1.0f),
        neighboringFoam * upward * 0.9f);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_fluid_integrate(
    uint3 dispatchThreadId : SV_DispatchThreadID
#if PM_TARGET_CUDA
    , uniform RWStructuredBuffer<PMPackedVec3> positions
    , uniform RWStructuredBuffer<PMPackedVec3> velocities
    , uniform StructuredBuffer<PMPackedVec3> accelerations
    , uniform RWStructuredBuffer<float> foam
    , uniform StructuredBuffer<uint> countBuffer
    , uniform PMFluidConstants constants
    , uniform RWStructuredBuffer<PMPackedVec3> previous
    , uniform StructuredBuffer<float> foamSources
#endif
)
{
    uint index = dispatchThreadId.x;
#if PM_TARGET_METAL
    PMFluidConstants constants = constantsBuffer;
    uint count = metadata[0].count;
#else
    uint count = countBuffer[0];
#endif
    if (index >= count)
        return;
    previous[index] = positions[index];
    float3 velocity = pmLoad(velocities[index]) +
        (pmLoad(constants.gravity) + pmLoad(accelerations[index])) *
            constants.timestep;
    velocity *= exp(-constants.velocityDamping * constants.timestep);
    velocity = pmLimit(velocity, constants.maximumSpeed);
    float3 position = pmLoad(positions[index]) +
                      velocity * constants.timestep;
    if (all(isfinite(position)) && all(isfinite(velocity)))
    {
        positions[index] = pmStore(position);
        velocities[index] = pmStore(velocity);
    }
    foam[index] = max(max(0.0f, foam[index] - constants.timestep * 0.7f),
                      foamSources[index]);
}
