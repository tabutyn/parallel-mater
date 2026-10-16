#include "dispatch_entry.slang"
// SPDX-License-Identifier: MIT
#include "dispatch_shape.slangh"
// Shared scene-query kernels.

import types;
import math;
import device_intrinsics;
import graphics_resources;

Vec3 hit_box_local_point(HitBox box, Vec3 point)
{
    return inverse_rotate(box.orientation, subtract(point, box.center));
}

bool hit_box_contains_point(HitBox box, Vec3 point)
{
    Vec3 local = hit_box_local_point(box, point);
    return abs(local.x) <= box.half_extents.x &&
           abs(local.y) <= box.half_extents.y &&
           abs(local.z) <= box.half_extents.z;
}

Vec3 hit_box_basis_axis(uint axis)
{
    if (axis == 0u) return {1.0f, 0.0f, 0.0f};
    if (axis == 1u) return {0.0f, 1.0f, 0.0f};
    return {0.0f, 0.0f, 1.0f};
}

bool hit_box_separates_triangle(
    Vec3 axis, Vec3 first, Vec3 second, Vec3 third, Vec3 half_extents)
{
    if (length_squared(axis) <= kEpsilon * kEpsilon) return false;
    float firstProjection = dot(axis, first);
    float secondProjection = dot(axis, second);
    float thirdProjection = dot(axis, third);
    float minimum = min(firstProjection, min(secondProjection, thirdProjection));
    float maximum = max(firstProjection, max(secondProjection, thirdProjection));
    float radius = abs(axis.x) * half_extents.x +
        abs(axis.y) * half_extents.y + abs(axis.z) * half_extents.z;
    return minimum > radius || maximum < -radius;
}

bool hit_box_overlaps_triangle(
    Vec3 first, Vec3 second, Vec3 third, Vec3 half_extents)
{
    Vec3 minimum = component_min(first, component_min(second, third));
    Vec3 maximum = component_max(first, component_max(second, third));
    if (minimum.x > half_extents.x || maximum.x < -half_extents.x ||
        minimum.y > half_extents.y || maximum.y < -half_extents.y ||
        minimum.z > half_extents.z || maximum.z < -half_extents.z) return false;
    Vec3 edges[3] = {
        subtract(second, first), subtract(third, second), subtract(first, third)};
    if (hit_box_separates_triangle(cross(edges[0], edges[1]),
        first, second, third, half_extents)) return false;
    for (uint edge = 0u; edge < 3u; ++edge)
        for (uint axis = 0u; axis < 3u; ++axis)
            if (hit_box_separates_triangle(cross(edges[edge],
                hit_box_basis_axis(axis)), first, second, third, half_extents))
                return false;
    return true;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void query_rigid_hit_box_kernel(
    uniform HitBox box, uniform PMBuffer<BodyParameters> parameters,
    uniform PMBuffer<RigidBodyState> states, uniform PMBuffer<TriangleMeshResource> meshes,
    uniform uint body_count, uniform PMByteBuffer matches PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint body = pmThreadIndex();
    if (body >= body_count) return;
    TriangleMeshResource mesh = meshes[parameters[body].mesh.index];
    RigidBodyState state = states[body];
    bool overlap = false;
    for (uint index = 0u; index < mesh.index_count; index += 3u)
    {
        Vec3 first = hit_box_local_point(box,
            transform_point(state, mesh.vertices[mesh.indices[index]]));
        Vec3 second = hit_box_local_point(box,
            transform_point(state, mesh.vertices[mesh.indices[index + 1u]]));
        Vec3 third = hit_box_local_point(box,
            transform_point(state, mesh.vertices[mesh.indices[index + 2u]]));
        if (hit_box_overlaps_triangle(first, second, third, box.half_extents))
        {
            overlap = true;
            break;
        }
    }
    if (!overlap && pmBufferPresent(mesh.solid_planes))
    {
        Vec3 center = inverse_rotate(state.orientation,
            subtract(box.center, state.position));
        overlap = true;
        for (uint triangle = 0u; triangle < mesh.index_count / 3u; ++triangle)
        {
            CollisionPlane plane = mesh.solid_planes[triangle];
            if (dot(plane.normal, center) > plane.offset + kEpsilon)
            {
                overlap = false;
                break;
            }
        }
    }
    matches[body] = overlap ? PMResourceUint8(1u) : PMResourceUint8(0u);
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void query_particle_hit_box_kernel(
    uniform HitBox box, uniform PMBuffer<Vec3> positions,
    uniform uint particle_count, uniform PMByteBuffer matches PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (particle >= particle_count) return;
    matches[particle] = hit_box_contains_point(box, positions[particle])
        ? PMResourceUint8(1u) : PMResourceUint8(0u);
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void export_fluid_particles(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> accelerations, uniform PMBuffer<float> temperatures,
    uniform PMBuffer<float> foam, uniform PMBuffer<uint> stable_ids,
    uniform uint particle_count, uniform uint capacity, uniform uint64_t revision,
    uniform PMBuffer<PortableDomainOutputHeader> header,
    uniform PMBuffer<PortableParticleState> destination PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (particle == 0u)
    {
        PortableDomainOutputHeader outputHeader;
        outputHeader.active_count = particle_count;
        outputHeader.capacity = capacity;
        outputHeader.revision = revision;
        header[0] = outputHeader;
    }
    if (particle >= particle_count) return;
    PortableParticleState output;
    output.position = positions[particle];
    output.velocity = velocities[particle];
    output.acceleration = accelerations[particle];
    output.temperature_celsius = temperatures[particle];
    output.foam = foam[particle];
    output.stable_id = stable_ids[particle];
    destination[particle] = output;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void export_cloth_vertices(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> rigid_forces, uniform PMBuffer<Vec3> liquid_forces,
    uniform PMBuffer<Vec3> soft_body_forces, uniform PMBuffer<Vec3> rope_forces,
    uniform uint count, uniform uint capacity, uniform uint64_t revision,
    uniform PMBuffer<PortableDomainOutputHeader> header,
    uniform PMBuffer<PortableClothVertexState> destination PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index == 0u)
    {
        PortableDomainOutputHeader outputHeader;
        outputHeader.active_count = count;
        outputHeader.capacity = capacity;
        outputHeader.revision = revision;
        header[0] = outputHeader;
    }
    if (index >= count) return;
    PortableClothVertexState output;
    output.position = positions[index];
    output.velocity = velocities[index];
    output.rigid_contact_force = rigid_forces[index];
    output.liquid_contact_force = liquid_forces[index];
    output.soft_body_contact_force = soft_body_forces[index];
    output.rope_contact_force = rope_forces[index];
    destination[index] = output;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void export_surface_positions(
    uniform PMBuffer<Vec3> positions, uniform uint count, uniform uint capacity,
    uniform uint64_t revision, uniform PMBuffer<PortableDomainOutputHeader> header,
    uniform PMBuffer<Vec3> destination PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index == 0u)
    {
        PortableDomainOutputHeader outputHeader;
        outputHeader.active_count = count;
        outputHeader.capacity = capacity;
        outputHeader.revision = revision;
        header[0] = outputHeader;
    }
    if (index < count) destination[index] = positions[index];
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void export_rope_nodes(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> constraint_forces, uniform PMBuffer<Vec3> contact_forces,
    uniform PMBuffer<Vec3> liquid_forces, uniform PMBuffer<Vec3> soft_body_forces,
    uniform uint count, uniform uint capacity, uniform uint64_t revision,
    uniform PMBuffer<PortableDomainOutputHeader> header,
    uniform PMBuffer<PortableRopeNodeState> destination PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index == 0u)
    {
        PortableDomainOutputHeader outputHeader;
        outputHeader.active_count = count;
        outputHeader.capacity = capacity;
        outputHeader.revision = revision;
        header[0] = outputHeader;
    }
    if (index >= count) return;
    PortableRopeNodeState output;
    output.position = positions[index];
    output.velocity = velocities[index];
    output.constraint_force = constraint_forces[index];
    output.contact_force = contact_forces[index];
    output.liquid_contact_force = liquid_forces[index];
    output.soft_body_contact_force = soft_body_forces[index];
    destination[index] = output;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void export_soft_body_nodes(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> rigid_forces, uniform PMBuffer<Vec3> cloth_forces,
    uniform PMBuffer<Vec3> liquid_forces, uniform PMBuffer<Vec3> rope_forces,
    uniform uint count, uniform uint capacity, uniform uint64_t revision,
    uniform PMBuffer<PortableDomainOutputHeader> header,
    uniform PMBuffer<PortableSoftBodyNodeState> destination PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index == 0u)
    {
        PortableDomainOutputHeader outputHeader;
        outputHeader.active_count = count;
        outputHeader.capacity = capacity;
        outputHeader.revision = revision;
        header[0] = outputHeader;
    }
    if (index >= count) return;
    PortableSoftBodyNodeState output;
    output.position = positions[index];
    output.velocity = velocities[index];
    output.rigid_contact_force = rigid_forces[index];
    output.cloth_contact_force = cloth_forces[index];
    output.liquid_contact_force = liquid_forces[index];
    output.rope_contact_force = rope_forces[index];
    destination[index] = output;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void export_smoke_particles(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<float> ages, uniform PMBuffer<float> number_densities,
    uniform PMBuffer<float> pressures, uniform PMBuffer<Vec3> vorticities,
    uniform uint count, uniform uint capacity, uniform uint64_t revision,
    uniform PMBuffer<PortableDomainOutputHeader> header,
    uniform PMBuffer<PortableSmokeParticleState> destination PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index == 0u)
    {
        PortableDomainOutputHeader outputHeader;
        outputHeader.active_count = count;
        outputHeader.capacity = capacity;
        outputHeader.revision = revision;
        header[0] = outputHeader;
    }
    if (index >= count) return;
    PortableSmokeParticleState output;
    output.position = positions[index];
    output.velocity = velocities[index];
    output.age = ages[index];
    output.number_density = number_densities[index];
    output.pressure = pressures[index];
    output.vorticity = vorticities[index];
    destination[index] = output;
}
