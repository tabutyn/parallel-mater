#include "dispatch_entry.slang"
// SPDX-License-Identifier: MIT
// Batched liquid cross-system exchange for realtime-single-step.

import types;
import math;
import device_intrinsics;
import graphics_resources;

static const uint kLiquidCouplingProfileVersion = 1u;
static const float kLiquidCouplingTimestep = 1.0f / 60.0f;
static const uint kLiquidCouplingSweeps = 4u;
static const uint kLiquidCouplingLiquid = 1u;
static const uint kLiquidCouplingRigid = 2u;
static const uint kLiquidCouplingCloth = 3u;
static const uint kLiquidCouplingSoftBody = 4u;
static const uint kLiquidCouplingRope = 5u;
static const uint kLiquidCouplingSmokeParticle = 6u;
static const uint kLiquidCouplingSmokeGrid = 7u;
static const uint kLiquidCouplingContact = 1u;
static const uint kLiquidCouplingContainment = 2u;
static const uint kLiquidCouplingPressure = 3u;
static const uint kLiquidCouplingViscosity = 4u;
static const uint kLiquidCouplingDrag = 5u;
static const uint kLiquidCouplingThermal = 6u;
static const uint kLiquidCouplingPhase = 7u;
static const uint kLiquidCouplingInvalidProfile = 1u << 22u;
static const uint kLiquidCouplingInvalidTimestep = 1u << 23u;
static const uint kLiquidCouplingInvalidMaterial = 1u << 24u;
static const uint kLiquidCouplingNonfiniteState = 1u << 25u;
static const uint kLiquidCouplingCorrectionExceeded = 1u << 26u;
static const uint kLiquidCouplingCapacityExceeded = 1u << 27u;

bool liquid_coupling_finite_vec3(Vec3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

bool liquid_coupling_profile_valid(uint profile, float dt, uint sweep)
{
    return profile == kLiquidCouplingProfileVersion && isfinite(dt) &&
        abs(dt - kLiquidCouplingTimestep) <= 1.0e-7f &&
        sweep < kLiquidCouplingSweeps;
}

bool liquid_coupling_mode_valid(uint family, uint mode)
{
    if (family == kLiquidCouplingLiquid)
        return mode == kLiquidCouplingContact ||
            mode == kLiquidCouplingPressure ||
            mode == kLiquidCouplingViscosity;
    if (family == kLiquidCouplingRigid || family == kLiquidCouplingCloth)
        return mode == kLiquidCouplingContact ||
            mode == kLiquidCouplingContainment;
    if (family == kLiquidCouplingSoftBody || family == kLiquidCouplingRope)
        return mode == kLiquidCouplingContact;
    if (family == kLiquidCouplingSmokeParticle ||
        family == kLiquidCouplingSmokeGrid)
        return mode == kLiquidCouplingDrag ||
            mode == kLiquidCouplingThermal || mode == kLiquidCouplingPhase;
    return false;
}

bool liquid_coupling_two_way(uint family, uint mode)
{
    return family != kLiquidCouplingSmokeParticle &&
        family != kLiquidCouplingSmokeGrid &&
        mode != kLiquidCouplingThermal && mode != kLiquidCouplingPhase;
}

void liquid_coupling_record_profile_invalid(
    uint profile, float dt, uint sweep, PMBuffer<uint> status)
{
    if (profile != kLiquidCouplingProfileVersion ||
        sweep >= kLiquidCouplingSweeps)
        pmAtomicOr(status, kLiquidCouplingInvalidProfile);
    if (!isfinite(dt) || abs(dt - kLiquidCouplingTimestep) > 1.0e-7f)
        pmAtomicOr(status, kLiquidCouplingInvalidTimestep);
}

// Constraint records come from the one shared F8-10 spatial publication and
// are reused for all four sweeps. This kernel never sorts or subdivides pairs.
[shader("compute")]
[numthreads(128, 1, 1)]
void liquid_coupling_project(
    uniform PMBuffer<Vec3> liquid_positions,
    uniform PMBuffer<Vec3> liquid_previous_positions,
    uniform PMBuffer<Vec3> liquid_velocities,
    uniform PMBuffer<float> liquid_inverse_masses,
    uniform PMBuffer<float> liquid_temperatures,
    uniform PMBuffer<uint> liquid_generations, uniform uint liquid_count,
    uniform PMBuffer<Vec3> partner_positions,
    uniform PMBuffer<Vec3> partner_previous_positions,
    uniform PMBuffer<Vec3> partner_velocities,
    uniform PMBuffer<float> partner_inverse_masses,
    uniform PMBuffer<Vec3> partner_angular_velocities,
    uniform PMBuffer<Vec3> partner_inverse_inertias,
    uniform PMBuffer<float> partner_temperatures,
    uniform PMBuffer<uint> partner_generations, uniform uint partner_count,
    uniform PMBuffer<uint> constraint_liquids,
    uniform PMBuffer<uint> constraint_partners,
    uniform PMBuffer<uint> constraint_liquid_generations,
    uniform PMBuffer<uint> constraint_partner_generations,
    uniform PMBuffer<uint> constraint_families,
    uniform PMBuffer<uint> constraint_modes,
    uniform PMBuffer<Vec3> constraint_normals,
    uniform PMBuffer<Vec3> constraint_partner_arms,
    uniform PMBuffer<float> constraint_contact_distances,
    uniform PMBuffer<float> constraint_frictions,
    uniform PMBuffer<float> constraint_responses,
    uniform PMBuffer<float> constraint_transfer_masses,
    uniform PMByteBuffer constraint_material_equal,
    uniform PMByteBuffer constraint_phase_capacity,
    uniform PMByteBuffer constraint_active, uniform uint constraint_count,
    uniform float characteristic_length, uniform float maximum_speed,
    uniform float dt, uniform uint profile_version, uniform uint sweep,
    uniform PMBuffer<Vec3> liquid_position_deltas,
    uniform PMBuffer<Vec3> partner_position_deltas,
    uniform PMBuffer<Vec3> liquid_impulses,
    uniform PMBuffer<Vec3> partner_angular_impulses,
    uniform PMBuffer<float> liquid_temperature_deltas,
    uniform PMBuffer<float> phase_transfer_masses,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint constraint = pmThreadIndex();
    if (!liquid_coupling_profile_valid(profile_version, dt, sweep))
    {
        if (constraint == 0u)
            liquid_coupling_record_profile_invalid(
                profile_version, dt, sweep, status);
        return;
    }
    if (constraint >= constraint_count) return;
    liquid_position_deltas[constraint] = {};
    partner_position_deltas[constraint] = {};
    liquid_impulses[constraint] = {};
    partner_angular_impulses[constraint] = {};
    liquid_temperature_deltas[constraint] = 0.0f;
    phase_transfer_masses[constraint] = 0.0f;
    if (constraint_active[constraint] == 0u) return;

    uint liquid = constraint_liquids[constraint];
    uint partner = constraint_partners[constraint];
    if (liquid >= liquid_count || partner >= partner_count)
    {
        pmAtomicOr(status, kLiquidCouplingInvalidMaterial);
        return;
    }
    if (liquid_generations[liquid] !=
            constraint_liquid_generations[constraint] ||
        partner_generations[partner] !=
            constraint_partner_generations[constraint])
        return;

    uint family = constraint_families[constraint];
    uint mode = constraint_modes[constraint];
    float liquidInverseMass = liquid_inverse_masses[liquid];
    float partnerInverseMass = partner_inverse_masses[partner];
    bool twoWay = liquid_coupling_two_way(family, mode);
    float massRatio = liquidInverseMass > 0.0f && partnerInverseMass > 0.0f
        ? max(liquidInverseMass, partnerInverseMass) /
            min(liquidInverseMass, partnerInverseMass) : 1.0f;
    Vec3 liquidPosition = liquid_positions[liquid];
    Vec3 partnerPosition = partner_positions[partner];
    Vec3 liquidVelocity = liquid_velocities[liquid];
    Vec3 partnerVelocity = partner_velocities[partner];
    Vec3 arm = constraint_partner_arms[constraint];
    if (family == kLiquidCouplingRigid)
        partnerVelocity = add(partnerVelocity,
            cross(partner_angular_velocities[partner], arm));
    Vec3 relativeVelocity = subtract(liquidVelocity, partnerVelocity);
    bool baseValid = liquid_coupling_mode_valid(family, mode) &&
        isfinite(characteristic_length) && characteristic_length >= 0.005f &&
        characteristic_length <= 2.0f && isfinite(maximum_speed) &&
        maximum_speed >= 0.0f && maximum_speed <= 20.0f &&
        maximum_speed * dt <= 0.5f * characteristic_length &&
        isfinite(liquidInverseMass) && liquidInverseMass >= 0.0001f &&
        liquidInverseMass <= 1000.0f && isfinite(partnerInverseMass) &&
        ((twoWay && partnerInverseMass >= 0.0001f &&
            partnerInverseMass <= 1000.0f && massRatio <= 100.0f) ||
         (!twoWay && partnerInverseMass == 0.0f)) &&
        (family != kLiquidCouplingLiquid ||
            constraint_material_equal[constraint] != 0u) &&
        liquid_coupling_finite_vec3(liquidPosition) &&
        liquid_coupling_finite_vec3(liquid_previous_positions[liquid]) &&
        liquid_coupling_finite_vec3(liquidVelocity) &&
        liquid_coupling_finite_vec3(partnerPosition) &&
        liquid_coupling_finite_vec3(partner_previous_positions[partner]) &&
        liquid_coupling_finite_vec3(partnerVelocity) &&
        liquid_coupling_finite_vec3(arm) &&
        liquid_coupling_finite_vec3(relativeVelocity) &&
        abs(liquidPosition.x) <= 10000.0f &&
        abs(liquidPosition.y) <= 10000.0f &&
        abs(liquidPosition.z) <= 10000.0f;
    if (!baseValid)
    {
        pmAtomicOr(status, kLiquidCouplingInvalidMaterial);
        return;
    }

    float distance = constraint_contact_distances[constraint];
    float friction = constraint_frictions[constraint];
    float response = constraint_responses[constraint];
    Vec3 liquidCorrection = {};
    Vec3 liquidImpulse = {};
    if (mode == kLiquidCouplingContact ||
        mode == kLiquidCouplingContainment ||
        mode == kLiquidCouplingPressure)
    {
        Vec3 normal = constraint_normals[constraint];
        float normalLength = vector_length(normal);
        bool contactLike = mode != kLiquidCouplingPressure;
        bool valid = liquid_coupling_finite_vec3(normal) &&
            normalLength > 1.0e-6f && isfinite(distance) &&
            distance > 0.0f && distance <= characteristic_length &&
            isfinite(friction) && friction >= 0.0f && friction <= 2.0f &&
            ((contactLike && response == 0.0f) ||
             (!contactLike && friction == 0.0f && response >= 0.05f &&
                response <= 0.995f));
        if (!valid)
        {
            pmAtomicOr(status, kLiquidCouplingInvalidMaterial);
            return;
        }
        normal = multiply(normal, 1.0f / normalLength);
        float separation = dot(subtract(
            liquidPosition, add(partnerPosition, arm)), normal);
        float previousSeparation = dot(subtract(
            liquid_previous_positions[liquid],
            add(partner_previous_positions[partner], arm)), normal);
        if (!isfinite(separation) || !isfinite(previousSeparation))
        {
            pmAtomicOr(status, kLiquidCouplingNonfiniteState);
            return;
        }
        float penetration = max(0.0f, distance - separation);
        if (previousSeparation >= distance && separation < distance)
            penetration = max(penetration, distance - separation);
        float inverseMass = liquidInverseMass + partnerInverseMass;
        float correction = min(penetration *
            (contactLike ? 1.0f : response),
            0.25f * characteristic_length);
        liquidCorrection = multiply(normal,
            correction * liquidInverseMass / inverseMass);
        partner_position_deltas[constraint] = multiply(normal,
            -correction * partnerInverseMass / inverseMass);

        float angularInverseMass = 0.0f;
        if (family == kLiquidCouplingRigid)
        {
            Vec3 axis = cross(arm, normal);
            Vec3 angularVelocity = {
                axis.x * partner_inverse_inertias[partner].x,
                axis.y * partner_inverse_inertias[partner].y,
                axis.z * partner_inverse_inertias[partner].z};
            angularInverseMass = dot(cross(angularVelocity, arm), normal);
        }
        float effectiveInverseMass = inverseMass + angularInverseMass;
        float closingSpeed = dot(relativeVelocity, normal);
        float normalMagnitude = closingSpeed < 0.0f &&
            effectiveInverseMass > kEpsilon
            ? -closingSpeed / effectiveInverseMass : 0.0f;
        if (!contactLike && penetration > 0.0f)
            normalMagnitude += response * penetration /
                (dt * effectiveInverseMass);
        liquidImpulse = multiply(normal, normalMagnitude);
        Vec3 tangent = subtract(relativeVelocity,
            multiply(normal, closingSpeed));
        float tangentSpeed = vector_length(tangent);
        if (contactLike && tangentSpeed > 1.0e-6f && friction > 0.0f)
        {
            Vec3 tangentImpulse = multiply(tangent,
                -min(tangentSpeed / inverseMass,
                    friction * normalMagnitude) / tangentSpeed);
            liquidImpulse = add(liquidImpulse, tangentImpulse);
        }
    }
    else if (mode == kLiquidCouplingViscosity ||
             mode == kLiquidCouplingDrag)
    {
        bool valid = isfinite(distance) && distance > 0.0f &&
            distance <= characteristic_length && friction == 0.0f &&
            isfinite(response) && response >= 0.05f && response <= 0.995f;
        if (!valid)
        {
            pmAtomicOr(status, kLiquidCouplingInvalidMaterial);
            return;
        }
        float inverseMass = mode == kLiquidCouplingDrag
            ? liquidInverseMass : liquidInverseMass + partnerInverseMass;
        liquidImpulse = multiply(relativeVelocity, -response / inverseMass);
    }
    else if (mode == kLiquidCouplingThermal)
    {
        float liquidTemperature = liquid_temperatures[liquid];
        float heaterTemperature = partner_temperatures[partner];
        bool valid = distance == 0.0f && friction == 0.0f &&
            isfinite(response) && response >= 0.05f && response <= 0.995f &&
            isfinite(liquidTemperature) && isfinite(heaterTemperature) &&
            liquidTemperature >= -273.15f &&
            heaterTemperature >= -273.15f;
        if (!valid)
        {
            pmAtomicOr(status, kLiquidCouplingInvalidMaterial);
            return;
        }
        liquid_temperature_deltas[constraint] =
            max(0.0f, heaterTemperature - liquidTemperature) * response;
    }
    else
    {
        float transfer = constraint_transfer_masses[constraint];
        float liquidMass = 1.0f / liquidInverseMass;
        bool valid = distance == 0.0f && friction == 0.0f &&
            response == 0.0f && isfinite(transfer) && transfer > 0.0f &&
            transfer <= liquidMass;
        if (!valid)
        {
            pmAtomicOr(status, kLiquidCouplingInvalidMaterial);
            return;
        }
        if (constraint_phase_capacity[constraint] == 0u)
        {
            pmAtomicOr(status, kLiquidCouplingCapacityExceeded);
            return;
        }
        phase_transfer_masses[constraint] = transfer;
    }
    if (!liquid_coupling_finite_vec3(liquidCorrection) ||
        !liquid_coupling_finite_vec3(liquidImpulse))
    {
        pmAtomicOr(status, kLiquidCouplingNonfiniteState);
        return;
    }
    liquid_position_deltas[constraint] = liquidCorrection;
    liquid_impulses[constraint] = liquidImpulse;
    if (family == kLiquidCouplingRigid)
        partner_angular_impulses[constraint] = cross(
            arm, multiply(liquidImpulse, -1.0f));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void liquid_coupling_apply_particles(
    uniform PMBuffer<Vec3> liquid_positions,
    uniform PMBuffer<Vec3> liquid_velocities,
    uniform PMBuffer<float> liquid_inverse_masses,
    uniform PMBuffer<float> liquid_temperatures,
    uniform PMBuffer<float> liquid_correction_used,
    uniform PMBuffer<uint> liquid_generations, uniform uint liquid_count,
    uniform PMBuffer<uint> constraint_liquids,
    uniform PMBuffer<uint> constraint_liquid_generations,
    uniform PMByteBuffer constraint_active, uniform uint constraint_count,
    uniform PMBuffer<Vec3> liquid_position_deltas,
    uniform PMBuffer<Vec3> liquid_impulses,
    uniform PMBuffer<float> liquid_temperature_deltas,
    uniform float characteristic_length, uniform float maximum_speed,
    uniform float dt, uniform uint profile_version, uniform uint sweep,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint liquid = pmThreadIndex();
    if (!liquid_coupling_profile_valid(profile_version, dt, sweep))
    {
        if (liquid == 0u)
            liquid_coupling_record_profile_invalid(
                profile_version, dt, sweep, status);
        return;
    }
    if (liquid >= liquid_count) return;
    Vec3 correction = {}, impulse = {};
    float temperatureDelta = 0.0f;
    for (uint constraint = 0u; constraint < constraint_count; ++constraint)
    {
        if (constraint_active[constraint] == 0u ||
            constraint_liquids[constraint] != liquid ||
            constraint_liquid_generations[constraint] !=
                liquid_generations[liquid])
            continue;
        correction = add(correction, liquid_position_deltas[constraint]);
        impulse = add(impulse, liquid_impulses[constraint]);
        // Multiple heaters share one endpoint update. Taking the strongest
        // admissible proposal preserves monotonicity without summing several
        // individually bounded responses past their hottest endpoint.
        temperatureDelta = max(temperatureDelta,
            liquid_temperature_deltas[constraint]);
    }
    float allowance = max(0.0f,
        0.25f * characteristic_length - liquid_correction_used[liquid]);
    Vec3 applied = clamp_length(correction, allowance);
    if (vector_length(correction) > allowance + 1.0e-6f)
        pmAtomicOr(status, kLiquidCouplingCorrectionExceeded);
    liquid_correction_used[liquid] += vector_length(applied);
    Vec3 position = add(liquid_positions[liquid], applied);
    Vec3 velocity = clamp_length(add(liquid_velocities[liquid],
        multiply(impulse, liquid_inverse_masses[liquid])), maximum_speed);
    float temperature = liquid_temperatures[liquid] + temperatureDelta;
    if (!liquid_coupling_finite_vec3(position) ||
        !liquid_coupling_finite_vec3(velocity) || !isfinite(temperature))
    {
        pmAtomicOr(status, kLiquidCouplingNonfiniteState);
        return;
    }
    liquid_positions[liquid] = position;
    liquid_velocities[liquid] = velocity;
    liquid_temperatures[liquid] = temperature;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void liquid_coupling_apply_partners(
    uniform PMBuffer<Vec3> partner_positions,
    uniform PMBuffer<Vec3> partner_velocities,
    uniform PMBuffer<float> partner_inverse_masses,
    uniform PMBuffer<Vec3> partner_angular_velocities,
    uniform PMBuffer<Vec3> partner_inverse_inertias,
    uniform PMBuffer<float> partner_correction_used,
    uniform PMBuffer<uint> partner_generations, uniform uint partner_count,
    uniform PMBuffer<uint> constraint_partners,
    uniform PMBuffer<uint> constraint_partner_generations,
    uniform PMByteBuffer constraint_active, uniform uint constraint_count,
    uniform PMBuffer<Vec3> partner_position_deltas,
    uniform PMBuffer<Vec3> liquid_impulses,
    uniform PMBuffer<Vec3> partner_angular_impulses,
    uniform float characteristic_length, uniform float maximum_speed,
    uniform float maximum_angular_speed, uniform float dt,
    uniform uint profile_version, uniform uint sweep,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint partner = pmThreadIndex();
    if (!liquid_coupling_profile_valid(profile_version, dt, sweep))
    {
        if (partner == 0u)
            liquid_coupling_record_profile_invalid(
                profile_version, dt, sweep, status);
        return;
    }
    if (partner >= partner_count || partner_inverse_masses[partner] == 0.0f)
        return;
    Vec3 correction = {}, impulse = {}, angularImpulse = {};
    for (uint constraint = 0u; constraint < constraint_count; ++constraint)
    {
        if (constraint_active[constraint] == 0u ||
            constraint_partners[constraint] != partner ||
            constraint_partner_generations[constraint] !=
                partner_generations[partner])
            continue;
        correction = add(correction, partner_position_deltas[constraint]);
        impulse = subtract(impulse, liquid_impulses[constraint]);
        angularImpulse = add(angularImpulse,
            partner_angular_impulses[constraint]);
    }
    float allowance = max(0.0f,
        0.25f * characteristic_length - partner_correction_used[partner]);
    Vec3 applied = clamp_length(correction, allowance);
    if (vector_length(correction) > allowance + 1.0e-6f)
        pmAtomicOr(status, kLiquidCouplingCorrectionExceeded);
    partner_correction_used[partner] += vector_length(applied);
    Vec3 position = add(partner_positions[partner], applied);
    Vec3 velocity = clamp_length(add(partner_velocities[partner], multiply(
        impulse, partner_inverse_masses[partner])), maximum_speed);
    Vec3 angularVelocity = partner_angular_velocities[partner];
    angularVelocity = add(angularVelocity, {
        angularImpulse.x * partner_inverse_inertias[partner].x,
        angularImpulse.y * partner_inverse_inertias[partner].y,
        angularImpulse.z * partner_inverse_inertias[partner].z});
    angularVelocity = clamp_length(angularVelocity, maximum_angular_speed);
    if (!liquid_coupling_finite_vec3(position) ||
        !liquid_coupling_finite_vec3(velocity) ||
        !liquid_coupling_finite_vec3(angularVelocity))
    {
        pmAtomicOr(status, kLiquidCouplingNonfiniteState);
        return;
    }
    partner_positions[partner] = position;
    partner_velocities[partner] = velocity;
    partner_angular_velocities[partner] = angularVelocity;
}
